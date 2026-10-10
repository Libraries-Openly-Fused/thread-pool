#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace thread_pool {

// Thread pool with two ways of giving it work:
//  - submit(f, args...): one task, returns a std::future with its result.
//  - enqueue(task, count): a bulk task of `count` chunks. The workers take chunk indices from an atomic counter, so
//    no memory is allocated per chunk or per call, and task.complete() is called exactly once, by the last thread
//    that leaves the task, after every chunk has finished. The caller does not wait: completion is only signaled
//    through complete().
// Tasks are started in the order they were enqueued. Idle workers spin for a short time (setSpinTime) before going to
// sleep, so that work enqueued shortly after the previous one is picked up without waiting for the OS to wake them.
//
// A worker must not block waiting for work of its own pool: when every worker does, nobody is left to run that work.
// A thread that waits for the pool and may be a worker (isWorkerThread()) helps instead: wait(future) for submitted
// functions, runOneChunk() in its own wait loop otherwise.
class ThreadPool {
public:
  class BulkTask {
  public:
    virtual ~BulkTask() = default;
    // Executes chunk `chunk`, in [0, count). Called concurrently from several threads for different chunks. If it
    // throws, the exception is kept (see exception()) and the other chunks still run.
    virtual void run(std::size_t chunk) = 0;
    // Called once, after every chunk has finished, from the thread that finished last. Must not throw. The task may be
    // enqueued again from here, or after this has been called, but never while it is still in flight.
    virtual void complete() = 0;
    // The first exception thrown by run() in the current execution, or null. Meant to be read in complete().
    std::exception_ptr exception() const { return m_exception; }

  private:
    friend class ThreadPool;
    // All guarded by the pool mutex, except m_next, which the participants take chunks from
    std::atomic<std::size_t> m_next{0};
    std::size_t m_count{0};
    std::size_t m_participants{0};
    BulkTask* m_nextTask{nullptr}; // link of the pool's FIFO queue
    bool m_inFlight{false};
    std::atomic<bool> m_failed{false};
    std::exception_ptr m_exception; // written once, by the first chunk that throws
  };

private:
  // A submitted function, as a bulk task of one chunk that deletes itself when it is done. packaged_task stores what
  // the function throws in the future.
  template <typename R>
  class SingleTask final : public BulkTask {
  public:
    explicit SingleTask(std::packaged_task<R()>&& task) : m_task(std::move(task)) {}
    void run(std::size_t) override { m_task(); }
    void complete() override { delete this; }

  private:
    std::packaged_task<R()> m_task;
  };

  static const ThreadPool*& currentPool() {
    static thread_local const ThreadPool* pool = nullptr;
    return pool;
  }

  const std::size_t m_size;
  std::vector<std::thread> m_threads;          // guarded by m_mutex until joined
  std::mutex m_mutex;
  std::condition_variable m_condition;
  BulkTask* m_head{nullptr};                   // FIFO queue of tasks with chunks left to take, guarded by m_mutex
  BulkTask* m_tail{nullptr};
  std::atomic<std::size_t> m_queued{0};        // number of tasks in the queue, readable without the mutex
  std::size_t m_sleepers{0};                   // workers waiting on m_condition, guarded by m_mutex
  std::atomic<bool> m_shutdown{false};
  bool m_started{false};                       // guarded by m_mutex
  bool m_stopped{false};                       // guarded by m_mutex: shutdown() has joined the workers
  std::atomic<long long> m_spinNanoseconds{100000};

  static void pause() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
  }

  void pop() {
    BulkTask* task = m_head;
    m_head = task->m_nextTask;
    if (m_head == nullptr) {
      m_tail = nullptr;
    }
    task->m_nextTask = nullptr;
    m_queued.fetch_sub(1, std::memory_order_release);
  }

  static void runChunk(BulkTask& task, const std::size_t chunk) {
    try {
      task.run(chunk);
    } catch (...) {
      if (!task.m_failed.exchange(true, std::memory_order_relaxed)) {
        task.m_exception = std::current_exception();
      }
    }
  }

  // Joins the task at the head of the queue and runs up to maxChunks of its chunks, then leaves it, and completes it if
  // this was the last thread in it after all its chunks were taken. Called with the lock held, returns with it
  // released. Returns false if the queue was empty.
  bool participate(std::unique_lock<std::mutex>& lock, const std::size_t maxChunks) {
    BulkTask* task = m_head;
    if (task == nullptr) {
      lock.unlock();
      return false;
    }
    ++task->m_participants;
    const std::size_t count = task->m_count;
    lock.unlock();

    for (std::size_t done = 0; done < maxChunks; ++done) {
      const std::size_t chunk = task->m_next.fetch_add(1, std::memory_order_relaxed);
      if (chunk >= count) {
        break;
      }
      runChunk(*task, chunk);
    }

    lock.lock();
    // Once every chunk has been taken, no other thread may join the task
    const bool allTaken = task->m_next.load(std::memory_order_relaxed) >= count;
    if (allTaken && m_head == task) {
      pop();
    }
    // A participant only leaves after finishing the chunks it took, so when the last one leaves after all of them
    // were taken, every chunk has finished
    const bool last = --task->m_participants == 0 && allTaken;
    if (last) {
      task->m_inFlight = false;
    }
    lock.unlock();
    if (last) {
      task->complete();
    }
    return true;
  }

  void workerLoop() {
    currentPool() = this;
    std::unique_lock<std::mutex> lock(m_mutex, std::defer_lock);
    while (true) {
      // Spin for a while before sleeping, so that a new task is picked up with low latency
      const auto spinEnd = std::chrono::steady_clock::now() + std::chrono::nanoseconds(m_spinNanoseconds.load());
      while (m_queued.load(std::memory_order_acquire) == 0 && !m_shutdown.load(std::memory_order_acquire) &&
             std::chrono::steady_clock::now() < spinEnd) {
        for (int i = 0; i < 64; ++i) {
          pause();
        }
      }
      lock.lock();
      while (m_head == nullptr && !m_shutdown.load(std::memory_order_relaxed)) {
        ++m_sleepers;
        m_condition.wait(lock);
        --m_sleepers;
      }
      if (m_head == nullptr) {
        // Shutting down and every queued task has been started
        lock.unlock();
        return;
      }
      participate(lock, static_cast<std::size_t>(-1));
    }
  }

public:
  explicit ThreadPool(const int n_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency())))
      : m_size(static_cast<std::size_t>(std::max(1, n_threads))) {}

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  ~ThreadPool() { shutdown(); }

  // Starts the worker threads. Calling it more than once does nothing. If a thread cannot be created, the ones already
  // created are stopped (after running the tasks already enqueued) and the exception is rethrown; init() can then be
  // called again.
  void init() {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_started || m_stopped) {
      return;
    }
    std::vector<std::thread> threads;
    try {
      threads.reserve(m_size);
      for (std::size_t i = 0; i < m_size; ++i) {
        threads.emplace_back([this] { workerLoop(); });
      }
    } catch (...) {
      m_shutdown.store(true, std::memory_order_release);
      lock.unlock();
      m_condition.notify_all();
      for (auto& thread : threads) {
        thread.join();
      }
      lock.lock();
      m_shutdown.store(false, std::memory_order_release);
      throw;
    }
    m_threads = std::move(threads);
    m_started = true;
  }

  // Finishes every task already enqueued, then stops and joins the worker threads. Calling it more than once does
  // nothing. Enqueueing after it throws std::logic_error. Must not be called from a worker of this pool.
  void shutdown() {
    if (isWorkerThread()) {
      throw std::logic_error("ThreadPool::shutdown called from one of its own workers");
    }
    std::vector<std::thread> threads;
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_shutdown.store(true, std::memory_order_release);
      threads = std::move(m_threads);
    }
    m_condition.notify_all();
    for (auto& thread : threads) {
      thread.join();
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stopped = true;
  }

  std::size_t size() const { return m_size; }

  // True when called from one of the worker threads of this pool
  bool isWorkerThread() const { return currentPool() == this; }

  // How long an idle worker spins before sleeping (100 microseconds by default, 0 to sleep right away)
  void setSpinTime(const std::chrono::nanoseconds spin) { m_spinNanoseconds.store(spin.count()); }

  // Runs task.run(0) ... task.run(count - 1) on the workers, then task.complete(). Returns immediately and never
  // allocates. The task must stay alive until complete() is called, and must not be enqueued again before that
  // (std::logic_error).
  void enqueue(BulkTask& task, const std::size_t count) {
    bool wake = false;
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      if (m_stopped) {
        throw std::logic_error("ThreadPool::enqueue after shutdown");
      }
      if (task.m_inFlight) {
        throw std::logic_error("ThreadPool::enqueue of a task whose previous execution has not completed");
      }
      task.m_inFlight = true;
      task.m_next.store(0, std::memory_order_relaxed);
      task.m_count = count;
      task.m_participants = 0;
      task.m_failed.store(false, std::memory_order_relaxed);
      task.m_exception = nullptr;
      task.m_nextTask = nullptr;
      if (m_tail != nullptr) {
        m_tail->m_nextTask = &task;
      } else {
        m_head = &task;
      }
      m_tail = &task;
      m_queued.fetch_add(1, std::memory_order_release);
      wake = m_sleepers > 0;
    }
    if (wake) {
      m_condition.notify_all();
    }
  }

  // Runs one chunk of the task at the head of the queue on the calling thread (and completes the task if that was its
  // last one). Returns false if there was no task. Lets a thread that waits for work of this pool help instead of
  // blocking, which a worker of the pool must do to avoid deadlocks:
  //     while (!done()) { if (!pool.runOneChunk()) std::this_thread::yield(); }
  bool runOneChunk() {
    std::unique_lock<std::mutex> lock(m_mutex);
    return participate(lock, 1);
  }

  // Waits until `future` is ready. On a worker of this pool it runs chunks of queued tasks meanwhile, instead of
  // blocking the worker.
  template <typename T>
  void wait(const std::future<T>& future) {
    if (!isWorkerThread()) {
      future.wait();
      return;
    }
    while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      if (!runOneChunk()) {
        std::this_thread::yield();
      }
    }
  }

  // Submit a function to be executed asynchronously by the pool
  template <typename F, typename... Args>
  auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>> {
    using R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
    std::packaged_task<R()> packaged(
        [f = std::forward<F>(f), tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable -> R {
          return std::apply(f, std::move(tuple));
        });
    std::future<R> future = packaged.get_future();
    auto task = std::make_unique<SingleTask<R>>(std::move(packaged));
    enqueue(*task, 1);
    task.release(); // deletes itself in complete()
    return future;
  }
};

} // namespace thread_pool
