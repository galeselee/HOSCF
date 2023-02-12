#include <iostream>
#include <vector>
#include <sys/time.h>
#include <omp.h>
#include <cmath>
#include <algorithm>

using namespace std;

double wclock()
{
  struct timeval time;
  if (gettimeofday(&time,NULL)){
	            //  Handle error
		    std::cout << "Wall time handle error\n";
		    return 0.; 
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double CPUTIME(clock_t & t1,  const clock_t & t2)
{
  t1 = clock();
  return double(t1-t2)/CLOCKS_PER_SEC;
}



double WallTIME(double & t1,const double & t2)
{
  t1 = wclock(); 
  return t1-t2;
}


double calTest(int max_threads, vector<vector<double> > & a,  vector<vector<double> > & b, vector<vector<double> > &c)
{
	omp_set_num_threads(max_threads);
	int N = a.size();
	int N2 = N/100; 
	int n_sc = N/max_threads;

	clock_t t_cpu_begin = clock();
	clock_t t_cpu_end;
	double  t_w_begin = wclock();
	double  t_w_end;

#pragma omp parallel for 
	for(int i = 0; i< N; ++i){
		for(int k = 0; k < 10*N2; ++k)
		{	
			int j = k /10;
			c[i][j] = a[j][j]+b[i][j]-a[i][j]*b[i][j];//sqrt(i);
		}
	}
		
	double t_wall = WallTIME(t_w_end,t_w_begin);
	double t_cpu  = CPUTIME(t_cpu_end,t_cpu_begin);
	
	return t_wall;
}



int main(int argc, char **argv) { 
	int N = atoi(argv[1]);
	int N2 = N/100;
	vector<vector<double> > a(N,vector<double>(N2,1));
	vector<vector<double> > b(N,vector<double>(N2,2));
	vector<vector<double> > c(N,vector<double>(N2,0));
	
	cout << "n_threads\tt_wall\tspeedup_factor\tideal_fractor\n"; 
	int n = 7;
	vector<int> n_threads(n);
	vector<double> t_ratio(n,1);
	vector<double> t_all(n);
	for(int i = 0; i < n; ++i){
		n_threads[i] = pow(2,i);
		t_all[i] = calTest(n_threads[i],a,b,c);
		t_ratio[i] = t_all[0]/t_all[i];
		cout << n_threads[i] << "\t" << t_all[i] << "\t" << t_ratio[i] <<"\t" << pow(2,i) <<"\n";
	}

	
	
	return 0;
}
