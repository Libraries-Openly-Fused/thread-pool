// Tests of the thread pool: submit(), bulk tasks, completion, exceptions, misuse, helping and shutdown.
#include "../include/ThreadPool.h"

#include <atomic>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using thread_pool::ThreadPool;

static int failures = 0;
#define CHECK(cond)                                                                  \
  do {                                                                               \
    if (!(cond)) {                                                                   \
      std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                  \
      ++failures;                                                                    \
    }                                                                                \
  } while (0)

// Counts how many times each chunk runs and whether complete() came after all of them
class CountingTask : public ThreadPool::BulkTask {
public:
  explicit CountingTask(std::size_t count) : m_runs(count) {}
  void run(std::size_t chunk) override {
    m_runs[chunk].fetch_add(1);
    m_done.fetch_add(1);
  }
  void complete() override {
    m_doneAtComplete = m_done.load();
    // Notify while holding the lock: the waiter may destroy this task as soon as it sees the completion
    std::lock_guard<std::mutex> lock(m_mutex);
    m_completions++;
    m_condition.notify_all();
  }
  void wait() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_condition.wait(lock, [this] { return m_completions > 0; });
  }
  bool eachChunkOnce() const {
    for (const auto& r : m_runs) {
      if (r.load() != 1) return false;
    }
    return true;
  }
  std::size_t completions() {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_completions;
  }
  std::size_t doneAtComplete() const { return m_doneAtComplete; }

private:
  std::vector<std::atomic<int>> m_runs;
  std::atomic<std::size_t> m_done{0};
  std::size_t m_doneAtComplete{0};
  std::size_t m_completions{0};
  std::mutex m_mutex;
  std::condition_variable m_condition;
};

// Waits for a condition with a timeout, so that a deadlock fails the test instead of hanging it
template <typename Pred>
bool waitFor(Pred done, std::chrono::seconds timeout = std::chrono::seconds(20)) {
  const auto end = std::chrono::steady_clock::now() + timeout;
  while (!done()) {
    if (std::chrono::steady_clock::now() > end) return false;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return true;
}

// Chunks wait until released, so that the task stays in flight
class GatedTask : public ThreadPool::BulkTask {
public:
  std::atomic<bool> open{false};
  std::atomic<int> runs{0}, completions{0};
  void run(std::size_t) override {
    while (!open.load()) std::this_thread::yield();
    runs.fetch_add(1);
  }
  void complete() override { completions.fetch_add(1); }
};

// Chunk 3 throws; complete() records whether the exception reached it
class ThrowingTask : public ThreadPool::BulkTask {
public:
  std::atomic<int> runs{0}, completions{0};
  std::atomic<bool> sawException{false};
  void run(std::size_t chunk) override {
    runs.fetch_add(1);
    if (chunk == 3) throw std::runtime_error("bad chunk");
  }
  void complete() override {
    if (exception()) {
      try {
        std::rethrow_exception(exception());
      } catch (const std::runtime_error&) {
        sawException.store(true);
      }
    }
    completions.fetch_add(1);
  }
};

// Enqueues itself again from complete() until it has run `times` times
class RepeatingTask : public ThreadPool::BulkTask {
public:
  RepeatingTask(ThreadPool& pool, int times) : m_pool(pool), m_left(times) {}
  std::atomic<int> runs{0}, completions{0};
  void run(std::size_t) override { runs.fetch_add(1); }
  void complete() override {
    const int left = --m_left;
    if (left > 0) m_pool.enqueue(*this, 4);
    completions.fetch_add(1);
  }

private:
  ThreadPool& m_pool;
  std::atomic<int> m_left;
};

int multiply(int a, int b) { return a * b; }
void multiplyOutput(int& out, int a, int b) { out = a * b; }

int main() {
  {
    ThreadPool pool(4);
    pool.init();
    pool.init(); // second call does nothing
    CHECK(pool.size() == 4);

    // submit with a return value and with an output parameter
    auto f1 = pool.submit(multiply, 6, 7);
    int out = 0;
    auto f2 = pool.submit(multiplyOutput, std::ref(out), 5, 6);
    CHECK(f1.get() == 42);
    f2.get();
    CHECK(out == 30);

    // bulk task: every chunk exactly once, complete() once and after all of them
    for (std::size_t count : {0u, 1u, 3u, 1000u, 100000u}) {
      CountingTask task(count);
      pool.enqueue(task, count);
      task.wait();
      CHECK(task.eachChunkOnce());
      CHECK(task.completions() == 1);
      CHECK(task.doneAtComplete() == count);
    }

    // many bulk tasks in flight at the same time, with random sizes
    std::mt19937 gen(7);
    std::uniform_int_distribution<int> size(0, 500);
    std::vector<std::unique_ptr<CountingTask>> tasks;
    for (int i = 0; i < 2000; ++i) {
      const std::size_t count = static_cast<std::size_t>(size(gen));
      tasks.push_back(std::make_unique<CountingTask>(count));
      pool.enqueue(*tasks.back(), count);
    }
    for (auto& task : tasks) {
      task->wait();
      CHECK(task->eachChunkOnce());
      CHECK(task->completions() == 1);
    }

    // no spinning: workers sleep right away and are woken up for each task
    pool.setSpinTime(std::chrono::nanoseconds(0));
    for (int i = 0; i < 200; ++i) {
      CountingTask task(64);
      pool.enqueue(task, 64);
      task.wait();
      CHECK(task.eachChunkOnce());
    }
  }

  // misuse and exceptions
  {
    ThreadPool pool(4);
    pool.init();
    CHECK(!pool.isWorkerThread());
    CHECK(pool.submit([&pool] { return pool.isWorkerThread(); }).get());

    // enqueueing a task that is still in flight throws, and does not disturb it
    GatedTask gated;
    pool.enqueue(gated, 16);
    bool threw = false;
    try {
      pool.enqueue(gated, 16);
    } catch (const std::logic_error&) {
      threw = true;
    }
    CHECK(threw);
    gated.open.store(true);
    CHECK(waitFor([&] { return gated.completions.load() == 1; }));
    CHECK(gated.runs.load() == 16);
    // after complete() it can be enqueued again
    pool.enqueue(gated, 8);
    CHECK(waitFor([&] { return gated.completions.load() == 2; }));
    CHECK(gated.runs.load() == 24);

    // and it can be enqueued again from its own complete()
    RepeatingTask repeating(pool, 50);
    pool.enqueue(repeating, 4);
    CHECK(waitFor([&] { return repeating.completions.load() == 50; }));
    CHECK(repeating.runs.load() == 200);

    // an exception in a chunk reaches complete(), and the other chunks still run
    ThrowingTask throwing;
    pool.enqueue(throwing, 10);
    CHECK(waitFor([&] { return throwing.completions.load() == 1; }));
    CHECK(throwing.sawException.load());
    CHECK(throwing.runs.load() == 10);
    // the exception does not carry over to the next execution
    ThrowingTask clean;
    pool.enqueue(clean, 3);
    CHECK(waitFor([&] { return clean.completions.load() == 1; }));
    CHECK(!clean.sawException.load());

    // an exception in a submitted function goes to its future
    auto failing = pool.submit([]() -> int { throw std::runtime_error("submitted"); });
    threw = false;
    try {
      failing.get();
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw);

    // shutdown() from one of the workers throws instead of joining itself
    auto selfShutdown = pool.submit([&pool] { pool.shutdown(); });
    threw = false;
    try {
      selfShutdown.get();
    } catch (const std::logic_error&) {
      threw = true;
    }
    CHECK(threw);
  }

  // workers that wait for work of their own pool help with runOneChunk() instead of blocking: with more waiters than
  // workers this would deadlock otherwise
  {
    ThreadPool pool(2);
    pool.init();
    constexpr int WAITERS = 8;
    std::vector<std::future<int>> futures;
    for (int i = 0; i < WAITERS; ++i) {
      futures.push_back(pool.submit([&pool] {
        CountingTask inner(100);
        pool.enqueue(inner, 100);
        while (inner.completions() == 0) {
          if (!pool.runOneChunk()) std::this_thread::yield();
        }
        return inner.eachChunkOnce() ? 1 : 0;
      }));
    }
    int ok = 0;
    bool finished = true;
    for (auto& f : futures) {
      if (f.wait_for(std::chrono::seconds(20)) != std::future_status::ready) {
        finished = false;
        break;
      }
      ok += f.get();
    }
    CHECK(finished);
    if (!finished) {
      std::printf("FAILED: helping workers deadlocked\n");
      std::fflush(stdout);
      std::_Exit(1);
    }
    CHECK(ok == WAITERS);

    // the same with submit(): functions that submit more work and wait for it with wait(future)
    std::vector<std::future<int>> nested;
    for (int i = 0; i < WAITERS; ++i) {
      nested.push_back(pool.submit([&pool, i] {
        auto inner = pool.submit(multiply, i, 3);
        pool.wait(inner);
        return inner.get();
      }));
    }
    for (int i = 0; i < WAITERS && finished; ++i) {
      finished = nested[static_cast<std::size_t>(i)].wait_for(std::chrono::seconds(20)) == std::future_status::ready;
      if (finished) {
        CHECK(nested[static_cast<std::size_t>(i)].get() == 3 * i);
      }
    }
    CHECK(finished);
    if (!finished) {
      std::printf("FAILED: nested submit and wait deadlocked\n");
      std::fflush(stdout);
      std::_Exit(1);
    }
  }

  // shutdown finishes the tasks that were already enqueued
  {
    std::atomic<int> executed{0};
    std::vector<std::future<void>> futures;
    {
      ThreadPool pool(2);
      pool.init();
      for (int i = 0; i < 100; ++i) {
        futures.push_back(pool.submit([&executed] {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          executed.fetch_add(1);
        }));
      }
      pool.shutdown();
      pool.shutdown(); // a second call does nothing
      bool threw = false;
      try {
        pool.submit([] {});
      } catch (const std::logic_error&) {
        threw = true;
      }
      CHECK(threw);
    }
    CHECK(executed.load() == 100);
  }

  // destroying a pool without calling shutdown() joins its threads
  {
    ThreadPool pool(3);
    pool.init();
    CHECK(pool.submit(multiply, 2, 3).get() == 6);
  }

  std::printf(failures == 0 ? "All thread pool tests passed\n" : "%d thread pool checks failed\n", failures);
  return failures == 0 ? 0 : 1;
}
