#pragma once

#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>

// Thread safe implementation of a Queue using an std::queue
template <typename T>
class SafeQueue {
private:
  std::queue<T> m_queue;
  mutable std::mutex m_mutex;
public:
  SafeQueue() = default;

  SafeQueue(const SafeQueue& other) {
    std::lock_guard<std::mutex> lock(other.m_mutex);
    m_queue = other.m_queue;
  }

  SafeQueue& operator=(const SafeQueue&) = delete;

  bool empty() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.empty();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.size();
  }

  void enqueue(T t) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_queue.push(std::move(t));
  }

  bool dequeue(T& t) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_queue.empty()) {
      return false;
    }
    t = std::move(m_queue.front());
    m_queue.pop();
    return true;
  }
};
