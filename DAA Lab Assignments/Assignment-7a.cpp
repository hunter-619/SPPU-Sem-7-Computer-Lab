// Assignment 7 a
// Write a program to implement matrix multiplication. Also implement multithreaded matrix multiplication with either one thread per row or one thread per cell. Analyze and compare their performance.

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace std;
using namespace std::chrono;
using namespace std::chrono_literals;

const int threadSize = 16;
    
vector<vector<int>> matrixMultiply(vector<vector<int>> &A, vector<vector<int>> &B, int startRow, int endRow) {

    int rowA = A.size();
    int colA = A[0].size();
    int colB = B[0].size();

    vector<vector<int>> result(rowA, vector<int>(colB, 0));

    for(int i = startRow; i < endRow; i++) {
        for(int j = 0; j < colB; j++) {
            for(int k = 0; k < colA; k++) {
                result[i][j] += (A[i][j] * B[j][k]);
            }
        }
    }

    return result;
}

int main() {
    int rowA, colA, rowB, colB;
    cout << "Enter the size of matrix A: ";
    cin >> rowA >> colA;
    cout << "Enter the size of matrix B: ";
    cin >> rowB >> colB;

    vector<vector<int>> A(rowA, vector<int>(colA));
    vector<vector<int>> B(rowB, vector<int>(colB));
    cout << "Enter matrix A row wise: \n";
    for(int i = 0; i < rowA; i++)
        for(int j = 0; j < colA; j++)
            cin >> A[i][j];

    cout << "Enter matrix B row wise: \n";
    for(int i = 0; i < rowB; i++)
        for(int j = 0; j < colB; j++)
            cin >> B[i][j];

    auto start = high_resolution_clock::now();
    matrixMultiply(A, B, 0, rowA);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);

    cout << "Time taken by naive Matrix Multiplication: " << duration.count() << " nanoseconds\n";

    thread threads[threadSize];
    int rowPerThread = rowA / threadSize; 
    start = high_resolution_clock::now();
    for(int i = 0; i < threadSize; i++) {
        int startRow = i * rowPerThread;
        int endRow = (i == threadSize - 1)?rowA:(i + 1)*rowPerThread;

        threads[i] = thread(matrixMultiply, ref(A), ref(B), startRow, endRow);
    }

    for(int i = 0; i < threadSize; i++) {
        threads[i].join();
    }

    stop = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(stop - start);

    cout << "Time taken by multithread Matrix Multiplication: " << duration.count() << " nanoseconds\n";
    return 0;
}