// Assignment 7 b
// Implement merge sort and multithreaded merge sort. Compare time required by both the algorithms. Also analyze the performance of each algorithm for the best case and the worst case.

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

void mergeSortMultithread(std::vector<int>& arr, int left, int right, int depth) {
    if (depth == 0 || left >= right) {
        mergeSort(arr, left, right);
        return;
    }

    int mid = left + (right - left) / 2;

    std::thread leftThread(mergeSortMultithread, std::ref(arr), left, mid, depth - 1);
    std::thread rightThread(mergeSortMultithread, std::ref(arr), mid + 1, right, depth - 1);

    leftThread.join();
    rightThread.join();

    merge(arr, left, mid, right);
}

int main() {
    std::vector<int> arr = {12, 11, 13, 5, 6, 7};
    int arrSize = arr.size();

    auto startSimple = std::chrono::high_resolution_clock::now();
    mergeSort(arr, 0, arrSize - 1);
    auto endSimple = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationSimple = endSimple - startSimple;

    std::cout << "Sorted array (Simple Merge Sort): ";
    for (int num : arr)
        std::cout << num << " ";
    std::cout << std::endl;

    std::cout << "Time taken by Simple Merge Sort: " << durationSimple.count() << " seconds" << std::endl;

    arr = {12, 11, 13, 5, 6, 7};

    auto startMultithread = std::chrono::high_resolution_clock::now();
    mergeSortMultithread(arr, 0, arrSize - 1, 2); // Using 2 threads
    auto endMultithread = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationMultithread = endMultithread - startMultithread;

    std::cout << "Sorted array (Multithreaded Merge Sort): ";
    for (int num : arr)
        std::cout << num << " ";
    std::cout << std::endl;

    std::cout << "Time taken by Multithreaded Merge Sort: " << durationMultithread.count() << " seconds" << std::endl;

    return 0;
}
