// Assignment 6
// Design 8-Queens matrix having first Queen placed. Use backtracking to place remaining Queens to generate the final 8-queen’s matrix.

#include <iostream> 
#include <vector>

using namespace std;
void printSolution(vector<vector<int>> board, int N) 
{ 
    for (int i = 0; i < N; i++) { 
        for (int j = 0; j < N; j++)  {
            if(board[i][j]) {
                cout << "Q ";
            }
            else {
                cout << "_ ";
            }
        }
            
        cout << "\n"; 
    } 
    cout << "\n\n";
} 

bool isSafe(vector<vector<int>> board, int row, int col, int N) 
{ 
    int i, j; 
    for (i = 0; i < col; i++) 
        if (board[row][i]) 
            return false; 
    for (i = row, j = col; i >= 0 && j >= 0; i--, j--) 
        if (board[i][j]) 
            return false; 
    for (i = row, j = col; j >= 0 && i < N; i++, j--) 
        if (board[i][j]) 
            return false; 
  
    return true; 
} 

bool solveNQueen(vector<vector<int>> board, int col, int N) 
{ 
    if (col >= N) {
        printSolution(board, N);
    } 
   
    for (int i = 0; i < N; i++) { 
        if (isSafe(board, i, col, N)) { 
         
            board[i][col] = 1; 
         
            solveNQueen(board, col + 1, N);

            board[i][col] = 0; 
        } 
    } 
   
    return false; 
} 

int main() 
{ 
    int n;
    cout << "Enter size of board: ";
    cin >> n;

    vector<vector<int>> board(n, vector<int>(n, 0));
  
    solveNQueen(board, 0, n);

    return 0; 
}