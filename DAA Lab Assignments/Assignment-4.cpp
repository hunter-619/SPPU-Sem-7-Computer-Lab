// Assignment 4
// Write a program to solve a 0-1 Knapsack problem using dynamic programming or branch and bound strategy.

#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int dynamicKnapsack(vector<pair<int, int>> Items, int n, int weight) {
    vector<int> dp(weight + 1, 0);

    for(int i = 0; i < n; i++) {
        for(int j = weight; j >=0; j--) {
            if(Items[i].first <= j) {
                dp[j] = max(dp[j], dp[j - Items[i].first] + Items[i].second);
            }
        }
    }

    return dp[weight];
}

int main() {
     int n;
    cout << "Enter no of items: ";
    cin >> n;

    vector<pair<int, int>> Items(n);
    for(int i = 0; i < n; i++) {
        cout << "Enter Weight and Profit for each Item: ";
        cin>>Items[i].first>>Items[i].second;
    }

    int weight;
    cout << "Enter Capacity of Knapsack: ";
    cin >> weight;

    cout << "Maximum Profit Possible: " << dynamicKnapsack(Items, n, weight);
    return 0;
}