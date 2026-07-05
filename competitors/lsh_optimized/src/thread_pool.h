#pragma once

#include <pthread.h>
#include <vector>
#include <algorithm>

struct ThreadTask {
    int start_idx;
    int end_idx;
};

class ThreadPool {
private:
    int num_threads_;
    std::vector<pthread_t> threads_;
    std::vector<ThreadTask> tasks_;
    bool shutdown_ = false;

    pthread_mutex_t mutex_;
    pthread_cond_t cv_task_; // Condition variable to wake up workers
    pthread_cond_t cv_done_; // Condition variable to wake up main thread

    int active_tasks_ = 0;
    int completed_tasks_ = 0;

    void (*work_func_)(int, int, int, void*) = nullptr;
    void* user_data_ = nullptr;

    static void* worker_routine(void* arg) {
        ThreadPool* pool = static_cast<ThreadPool*>(arg);
        while (true) {
            pthread_mutex_lock(&pool->mutex_);
            while (pool->tasks_.empty() && !pool->shutdown_) {
                pthread_cond_wait(&pool->cv_task_, &pool->mutex_);
            }

            if (pool->shutdown_) {
                pthread_mutex_unlock(&pool->mutex_);
                break;
            }

            // Get a task from the list
            ThreadTask task = pool->tasks_.back();
            pool->tasks_.pop_back();
            int thread_id = pool->tasks_.size(); // Use size as a simple thread identifier (0 to num_threads_-1)
            pthread_mutex_unlock(&pool->mutex_);

            // Execute the work function
            if (pool->work_func_) {
                pool->work_func_(task.start_idx, task.end_idx, thread_id, pool->user_data_);
            }

            pthread_mutex_lock(&pool->mutex_);
            pool->completed_tasks_++;
            if (pool->completed_tasks_ == pool->active_tasks_) {
                pthread_cond_signal(&pool->cv_done_);
            }
            pthread_mutex_unlock(&pool->mutex_);
        }
        return nullptr;
    }

public:
    ThreadPool(int num_threads) : num_threads_(num_threads) {
        if (num_threads_ <= 0) num_threads_ = 1;
        pthread_mutex_init(&mutex_, nullptr);
        pthread_cond_init(&cv_task_, nullptr);
        pthread_cond_init(&cv_done_, nullptr);

        threads_.resize(num_threads_);
        for (int i = 0; i < num_threads_; ++i) {
            pthread_create(&threads_[i], nullptr, worker_routine, this);
        }
    }

    ~ThreadPool() {
        pthread_mutex_lock(&mutex_);
        shutdown_ = true;
        pthread_cond_broadcast(&cv_task_);
        pthread_mutex_unlock(&mutex_);

        for (int i = 0; i < num_threads_; ++i) {
            pthread_join(threads_[i], nullptr);
        }

        pthread_mutex_destroy(&mutex_);
        pthread_cond_destroy(&cv_task_);
        pthread_cond_destroy(&cv_done_);
    }

    int get_num_threads() const { return num_threads_; }

    void run_parallel(void (*work_func)(int, int, int, void*), void* user_data, int total_work_size) {
        if (total_work_size <= 0) return;

        pthread_mutex_lock(&mutex_);
        work_func_ = work_func;
        user_data_ = user_data;
        completed_tasks_ = 0;
        tasks_.clear();

        // Calculate chunk sizes and assign tasks
        int actual_threads = std::min(num_threads_, total_work_size);
        active_tasks_ = actual_threads;

        int chunk_size = total_work_size / actual_threads;
        int remainder = total_work_size % actual_threads;

        int start = 0;
        for (int i = 0; i < actual_threads; ++i) {
            int end = start + chunk_size + (i < remainder ? 1 : 0);
            tasks_.push_back({start, end});
            start = end;
        }

        // Wake up workers
        pthread_cond_broadcast(&cv_task_);

        // Block main thread until all workers finish their tasks
        while (completed_tasks_ < active_tasks_) {
            pthread_cond_wait(&cv_done_, &mutex_);
        }
        pthread_mutex_unlock(&mutex_);
    }
};
