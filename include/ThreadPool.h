#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

// Thread pool with two ways of giving it work:
//  - submit(f, args...): one task, returns a std::future with its result.
//  - enqueue(task, count): a bulk task of `count` chunks. The workers take chunk indices from an atomic counter, so
//    no memory is allocated per chunk or per call, and task.complete() is called exactly once, by the last worker
//    that leaves the task, after every chunk has finished. The caller does not wait: completion is only signaled
//    through complete().
// Tasks are started in the order they were enqueued. Idle workers spin for a short time (setSpinTime) before going to
// sleep, so that work enqueued shortly after the previous one is picked up without waiting for the OS to wake them.
class ThreadPool {
public:
  class BulkTask {
  public:
    virtual ~BulkTask() = default;
    // Executes chunk `chunk`, in [0, count). Called concurrently from several workers for different chunks.
    virtual void run(std::size_t chunk) = 0;
    // Called once, after every chunk has finished, from the worker that finished last.
    virtual void complete() = 0;

  private:
    friend class ThreadPool;
    std::atomic<std::size_t> m_next{0};
    std::size_t m_count{0};
    std::size_t m_participants{0}; // guarded by the pool mutex
  };

private:
  // A submitted function, as a bulk task of one chunk that deletes itself when it is done
  template <typename R>
  class SingleTask final : public BulkTask {
  public:
    explicit SingleTask(std::packaged_task<R()>&& task) : m_task(std::move(task)) {}
    void run(std::size_t) override { m_task(); }
    void complete() override { delete this; }

  private:
    std::packaged_task<R()> m_task;
  };

  std::vector<std::thread> m_threads;
  std::mutex m_mutex;
  std::condition_variable m_condition;
  std::deque<BulkTask*> m_tasks;            // guarded by m_mutex
  std::atomic<std::size_t> m_queued{0};     // m_tasks.size(), readable without the mutex while spinning
  std::size_t m_sleepers{0};                // workers waiting on m_condition, guarded by m_mutex
  std::atomic<bool> m_shutdown{false};
  std::atomic<long long> m_spinNanoseconds{100000};
  bool m_started{false};                    // guarded by m_mutex

  static void pause() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
  }

  void workerLoop() {
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
      while (m_tasks.empty() && !m_shutdown.load(std::memory_order_relaxed)) {
        ++m_sleepers;
        m_condition.wait(lock);
        --m_sleepers;
      }
      if (m_tasks.empty()) {
        // Shutting down and every queued task has been started
        return;
      }
      BulkTask* task = m_tasks.front();
      ++task->m_participants;
      lock.unlock();

      for (std::size_t chunk = task->m_next.fetch_add(1, std::memory_order_relaxed); chunk < task->m_count;
           chunk = task->m_next.fetch_add(1, std::memory_order_relaxed)) {
        task->run(chunk);
      }

      lock.lock();
      // Every chunk has been taken: no other worker may join this task any more
      if (!m_tasks.empty() && m_tasks.front() == task) {
        m_tasks.pop_front();
        m_queued.fetch_sub(1, std::memory_order_release);
      }
      // The participants that took chunks only leave after finishing them, so the last one to leave can complete it
      const bool last = --task->m_participants == 0;
      lock.unlock();
      if (last) {
        task->complete();
      }
    }
  }

public:
  explicit ThreadPool(const int n_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency())))
      : m_threads(static_cast<std::size_t>(std::max(1, n_threads))) {}

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  ~ThreadPool() { shutdown(); }

  // Starts the worker threads. Calling it more than once does nothing.
  void init() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_started) {
      return;
    }
    m_started = true;
    for (auto& thread : m_threads) {
      thread = std::thread([this] { workerLoop(); });
    }
  }

  // Finishes every task already enqueued, then stops and joins the worker threads. Calling it more than once does
  // nothing; tasks enqueued after it are not executed.
  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_shutdown.store(true, std::memory_order_release);
    }
    m_condition.notify_all();
    for (auto& thread : m_threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  std::size_t size() const { return m_threads.size(); }

  // How long an idle worker spins before sleeping (100 microseconds by default, 0 to sleep right away)
  void setSpinTime(const std::chrono::nanoseconds spin) { m_spinNanoseconds.store(spin.count()); }

  // Runs task.run(0) ... task.run(count - 1) on the workers, then task.complete(). Returns immediately. The task must
  // stay alive until complete() is called.
  void enqueue(BulkTask& task, const std::size_t count) {
    task.m_next.store(0, std::memory_order_relaxed);
    task.m_count = count;
    task.m_participants = 0;
    bool wake = false;
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_tasks.push_back(&task);
      m_queued.fetch_add(1, std::memory_order_release);
      wake = m_sleepers > 0;
    }
    if (wake) {
      m_condition.notify_all();
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
    enqueue(*new SingleTask<R>(std::move(packaged)), 1);
    return future;
  }
};
