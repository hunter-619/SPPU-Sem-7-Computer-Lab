// Assignment 2
// Implement job sequencing with deadlines using a greedy method.

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <queue>

using namespace std;

struct Job {
    string id;
    int dead;
    int profit;
};

class jobProfit {
    public:
    bool operator()(Job A, Job B) {
        return (A.profit < B.profit);
    }   
};

bool jobDeadline(Job A, Job B) { return (A.dead < B.dead); }

void jobschedule(vector<Job> Jobs, int n) {
    vector<Job> schedule;
    sort(Jobs.begin(), Jobs.end(), jobDeadline);
    
    priority_queue<Job, vector<Job>, jobProfit> jobQueue;

    for(int i = n-1; i >= 0; i--) {
        int slot_available = 0;

        if(i == 0) {
            slot_available = Jobs[i].dead;
        }
        else {
            slot_available = Jobs[i].dead - Jobs[i-1].dead;
        }

        jobQueue.push(Jobs[i]);

        if(slot_available > 0 && !jobQueue.empty()) {
            Job newJob = jobQueue.top();
            jobQueue.pop();

            slot_available--;
            schedule.push_back(newJob);
        }
    }

    sort(schedule.begin(), schedule.end(), jobDeadline);
    for(auto& item : schedule) {
        cout << item.id << ' ';
    }
    cout << endl;
    return;
}

int main() {
    int n;
    cout << "Enter the no. of jobs : ";
    cin >> n;

    vector<Job> Jobs(n);
    for(int i=0; i<n; i++) {
        cout << "Enter job details [Id Deadline Profit] : " << endl;
        cin >> Jobs[i].id;
        cin >> Jobs[i].dead;
        cin >> Jobs[i].profit;
    }

    cout << "Schedule :  ";
    jobschedule(Jobs, n);
}
