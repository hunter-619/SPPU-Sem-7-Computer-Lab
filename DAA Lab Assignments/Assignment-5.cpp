// Assignment 5
// Write a program to generate binomial coefficients using dynamic programming.

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int binomialCoefficient(int n, int k) {
    vector<int> C(k + 1, 0);
    C[0] = 1;

    for(int i = 1; i <= n; i++) {
        for(int j = min(i, k); j >0; j--) {
            C[j] = C[j] + C[j - 1];
        }
    }
    
    return C[k];
}

int main() {
    int n, k;
    cout << "Enter the value of N and K : ";
    cin >> n >> k;

    cout << "Value of C(" << n << "," << k << ")"<< " is " << binomialCoefficient(n, k);
    return 0;
}