// Assignment 7 c
// Implement the Naive string matching algorithm and Rabin-Karp algorithm for string matching. Observe difference in working of both the algorithms for the same input.

#include <iostream>

using namespace std;

bool naiveStringMatch(string text, string pat, int n, int m) {
    for(int i = 0; i < n - m; i++) {
        int j;
        for(int j = 0; j < m; j++) {
            if(text[i + j] != pat[j])
                break;

            if(j == m-1) {
                cout << "\n Pattern found at index " << i << "\n";
                return true;
            }
        }
    }

    return false;
}

bool rabinKarpStringMatch(string text, string pat, int n, int m) {
    int i, j;
    int p = 0;
    int t = 0;
    int h = 1;
    int d = 256;
    int q = INT_MAX;

    for(i = 0; i < m-1; i++)
        h = (h * d) % q;

    for(i = 0; i < m; i++) {
        p = (d * p + pat[i]) % q;
        t = (d * t + text[i]) % q;
    }

    for( i = 0; i <= n - m; i++) {
        if(p == t) {
            for(j = 0; j < m; j++) {
                if(text[i + j] != pat[j]) 
                    break;                
            }

            if(j == m){
                cout << "\n Pattern found at index " << i << "\n";
                return true;
            }
        }

        if(i < n -m) {
            t = (d * (t - text[i] * h) + text[i + m]) % q;

            if(t < 0)
                t += q;
        }
    }

    return false;
}

int main() {
    string a, b;
    cout << "Enter text: ";
    cin >> a;
    cout << "Enter pattern: ";
    cin >> b;

    int choice;
    do {
        cout << "\n ***Enter you choice*** \n";
        cout << "1. Naive String Matching \n";
        cout << "2. Rabin-Karp String Matching \n";
        cout << "3. Exit \n";
        cin >> choice;
        switch (choice)
        {
        case 1:
            if (!naiveStringMatch(a, b, a.length(), b.length())) {
                cout << "\n No match found \n";
            }
            break;
        case 2:
            if (!rabinKarpStringMatch(a, b, a.length(), b.length())) {
                cout << "\n No match found \n";
            }
        default:
            break;
        }
    }while(choice != 3);
    return 0;
}