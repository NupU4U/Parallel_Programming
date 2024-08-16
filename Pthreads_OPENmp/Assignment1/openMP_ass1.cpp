#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>

using namespace std;

int n;
int numOfThreads;
int residual;

mt19937_64 rng(random_device{}());
uniform_real_distribution<double> dist(10.0, 100.0);

void initialize_matrix_a(double** a, double** a_copy) {
	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			a[i][j]= dist(rng);
			a_copy[i][j]= a[i][j];
		}
	}
}

void initialize_matrix_l_u(double** l, double** u) {

	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			if(i>j) {
				l[i][j]= dist(rng);
			}
			else if(i<j) {
				u[i][j]= dist(rng);
			}
			else 
				l[i][j]=1.0;
			
		}
	}
}
void print_matrices(double** A, double** A_unmodified, double** P, double** L, double** U) {
	
    ofstream file;
    file.open("A.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            file << A[i][j] << " ";
        file << endl;
    }
    file.close();

    file.open("A_unmodified.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            file << A_unmodified[i][j] << " ";
        file << endl;
    }
    file.close();

    file.open("P.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            file << P[i][j] << " ";
        file << endl;
    }
    file.close();
    file.open("L.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            file << L[i][j] << " ";
        file << endl;
    }
    file.close();
    file.open("U.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            file << U[i][j] << " ";
        file << endl;
    }
    file.close();
}

void decomposition() {
	
	int* pi = (int*)malloc(n * sizeof(int));
   	double** P = (double**)malloc(n*sizeof(double*));
    double** A = (double**)malloc(n*sizeof(double*));
    double** L = (double**)malloc(n*sizeof(double*));
    double** U = (double**)malloc(n*sizeof(double*));
    double** A_unmodified = (double**)malloc(n*sizeof(double*));
    for (int i = 0; i < n; i++)
    {
        P[i] = (double*)malloc(n*sizeof(double));
        A[i] = (double*)malloc(n*sizeof(double));
        L[i] = (double*)malloc(n*sizeof(double));
        U[i] = (double*)malloc(n*sizeof(double));
        A_unmodified[i] = (double*)malloc(n*sizeof(double));
        for (int j = 0; j < n; j++)
        {
            P[i][j] = 0.0;
            A[i][j] = 0.0;
            L[i][j] = 0.0;
            U[i][j] = 0.0;
            A_unmodified[i][j] = 0.0;
        }
    }

    	initialize_matrix_a(A,A_unmodified);
    	initialize_matrix_l_u(L, U);
    	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			A_unmodified[i][j]=A[i][j];
		}
	}
	//start time
	auto start= std::chrono::high_resolution_clock::now();	
	for(int i=0; i<n; i++)
	{	
		pi[i]=i;
	}
	for(int k=0; k<n; k++) {
		// cout<<"k: "<<k<<endl;
		double maxVal=0.0;
		int k2= k;
		
		for(int i=k; i<n; i++) {
			if(abs(A[i][k])> maxVal) {
				maxVal= abs(A[i][k]);
				k2=i;
			}
		}
		if(maxVal==0) {
			cerr<<"Error Singular matrix encountered"<<endl;
			return;
		}
		
		swap(pi[k], pi[k2]);
		swap(A[k], A[k2]);
		
		for(int i=0; i<k; i++) {
			swap(L[k][i], L[k2][i]);
		}
		
		U[k][k]= A[k][k];
		
		for(int i=k+1; i<n; i++) {
			L[i][k] = A[i][k]/ U[k][k];
			U[k][i] = A[k][i];
		}
		int i;
		# pragma omp parallel for num_threads(numOfThreads) private(i) shared(A, L, U)
			for(i=k+1; i<n; i++) {
				for(int j=k+1; j<n; j++) {
					A[i][j]= A[i][j]-L[i][k]*U[k][j];
				}
			}
	}
	
	auto end= std::chrono::high_resolution_clock::now();
    double time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "Time taken by LU decomposition: " << time_taken << "ms\n";
	for(int i=0; i<n; i++) {
	P[i][pi[i]]=1.0;
	}
	// print_matrices(A, A_unmodified, P, L, U);
	if (residual == 1) {
		double l21_norm = 0.0;
		double residual = 0.0;
		cout<<"RESIDUAL MATRIX"<<endl;
		for(int i=0; i < n; i++) {
			for(int j=0; j < n; j++) {
				residual= 0.0;
				for(int k=0; k < n; k++) {
					residual += P[i][k] * A_unmodified[k][j] - L[i][k] * U[k][j] ;
				}
				l21_norm += residual * residual;
			}
		}
		cout << "L2,1 norm of the residual: " << l21_norm << endl;
	}
	//free memory
	delete [] (pi);
	free(P);
	free(A);
	free(A_unmodified);
	free(U);
	free(L);

}

int main(int argc, char* argv[]) {
	
	srand48((unsigned int) time(nullptr));
	n = atoi(argv[1]);
	numOfThreads = atoi(argv[2]);	
	residual = atoi(argv[3]);
	cout << "---------OpenMP---------\n";
	cout << "n: " << n << " threads: " << numOfThreads << endl;
	decomposition();
	return 0;
}
