#include<iostream> 
#include<pthread.h>
#include<random>
#include<fstream>
#include<sstream>
#include<chrono>
#include<cmath>
#include<ctime>

using namespace std;
int n;
int threads ; 
int residual;

mt19937_64 rng(random_device{}());
uniform_real_distribution<double> dist(10.0, 100.0);

void initialize_all(double** A, double** L, double** U, double** A_unmodified)
{
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++){
            A[i][j] = dist(rng);
            A_unmodified[i][j] = A[i][j];
        }
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            if(i <= j)
                U[i][j] = dist(rng);
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            if(i > j)
                L[i][j] = dist(rng);
            else if(i == j)
                L[i][j] = 1.0;
}

void* thread_result(void* input)
{
    double** A = static_cast<double**>(reinterpret_cast<void**>(input)[0]);
    double** L = static_cast<double**>(reinterpret_cast<void**>(input)[1]);
    double** U = static_cast<double**>(reinterpret_cast<void**>(input)[2]);
    int core = *static_cast<int*>(reinterpret_cast<void**>(input)[3]);
    int row = *static_cast<int*>(reinterpret_cast<void**>(input)[4]);
    for(int i=(row+1)+core*(n-(row+1))/threads; i < (row+1)+(core+1)*(n-(row+1))/threads; i++)
    {												
        for(int j=row+1; j < n; j++)
        {
            A[i][j] = A[i][j] -L[i][row]*U[row][j];  
        }
    }
    return nullptr;
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

void decomposition()
{
    // clock_t start = clock();
    auto start = std::chrono::high_resolution_clock::now();             
    int* pie  = (int*)malloc(n*sizeof( int)); 
    for ( int i = 0; i < n; i++){
        pie[i] = i;
    }
        // pie[i] = i;
    pthread_t arr_of_threads[threads];
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
    initialize_all(A, L, U, A_unmodified);
    // cout << "here\n";
    for ( int row =0 ; row < n; row++)
    {   
        // cout << "k: " << k << endl;
        int max_index = -2;
        double max = 0.0;
        for ( int i = row; i < n; i++)
        {
            if ( abs(A[i][row]) > max)
            {
                max = abs(A[i][row]);
                max_index = i;
            }
        }
        if(abs(max)< 0.000000001)                      
            perror("Error: not valid Matrix");

        swap(pie[row], pie[max_index]);
        swap(A[row], A[max_index]);

        for ( int j = 0 ; j < row; j++ ){
            swap(L[row][j],L[max_index][j]);
        }
        U[row][row] = A[row][row];
        for ( int i = row+1; i < n; i++)
        {
            L[i][row] = A[i][row]/(U[row][row]);
            U[row][i] = A[row][i];
        }

        for(int i = 0; i < threads; i++){
            // cout<<"i = "<<i<<"\n";
            void** in_args = new void*[5];
            in_args[0] = reinterpret_cast<void*>(A);
            in_args[1] = reinterpret_cast<void*>(L);
            in_args[2] = reinterpret_cast<void*>(U);
            in_args[3] = reinterpret_cast<void*>(new int(i)); 
            in_args[4] = reinterpret_cast<void*>(new int(row)); 

            pthread_create(&arr_of_threads[i], NULL, thread_result, reinterpret_cast<void*>(in_args));
        }
        for(int t = 0; t < threads; t++)
        {
            pthread_join(arr_of_threads[t], NULL);
        }
    }

    for (int i =0 ;i<n; i++)
    {
        P[i][pie[i]] = 1.0;
    }
    auto end= std::chrono::high_resolution_clock::now();
    double time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "Time taken by LU decomposition: " << time_taken << "ms\n";

    // print_matrices(A, A_unmodified, P, L, U);

    if (residual == 1)
    {
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

    free(pie);
    free(A);
    free(A_unmodified);
    free(L);
    free(U);
    free(P);
    // free(arr_of_threads);
}


int main(int argc, char* argv[])
{
    srand48((unsigned int) time(nullptr));
    n = atoi(argv[1]);
    threads = atoi(argv[2]);
    residual = atoi(argv[3]);
    cout << "---------Pthreads---------\n";
    cout << "n: " << n << " threads: " << threads << endl;
    decomposition();
    return 0;
}