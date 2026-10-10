// Tests of the thread pool: submit(), bulk tasks, completion, ordering of starts and shutdown.
#include "../include/ThreadPool.h"

#include <atomic>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

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
      const std::size_t count = size(gen);
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
