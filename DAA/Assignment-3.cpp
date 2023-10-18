// Asisgnment 3
// Write a program to solve a fractional Knapsack problem using a greedy method

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

double fracKnapsack(vector<pair<double, double>> Items, int n, int weight) {
    double maxProfit = 0;
    sort(Items.begin(), Items.end(), [](pair<int, int> A, pair<int, int> B){ return (A.second < B.second);});
    for(int i = 0; i < n; i++) {
        if(Items[i].first < weight) {
            maxProfit += Items[i].second;
            weight -= Items[i].first;
        }
        else {
            maxProfit += Items[i].second * (weight / Items[i].first);
            weight -= weight / Items[i].first;
        }
    }
    return maxProfit;
}

int main() {
    int n;
    cout << "Enter no of items: ";
    cin >> n;

    vector<pair<double, double>> Items(n);
    for(int i = 0; i < n; i++) {
        cout << "Enter Weight and Profit for each Item: ";
        cin>>Items[i].first>>Items[i].second;
    }

    int weight;
    cout << "Enter Capacity of Knapsack: ";
    cin >> weight;

    
    cout << "Maximum Profit Possible: " << fracKnapsack(Items, n, weight);
    return 0;
}