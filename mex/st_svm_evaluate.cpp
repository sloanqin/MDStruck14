/**
* @file st_svm_eval.cpp
* @mex interface for struct svm eval(st_svm_eval)
* @author yuanyuan qin
* @date 2017.3
*/

//#include "mexopencv.hpp"
#include "mex.h" 
#include <vector>
#include <cmath>
using namespace std;
//using namespace cv;

double st_svm_kernel_eval(vector<double> &x1, vector<double> &x2, double kernerlSigma)
{
	//res = exp(-m_sigma*squaredNorm(x1-x2));

	if (x1.size() != x2.size())
		mexErrMsgIdAndTxt("st_svm_kernel_eval:error", "vectors' dimension are inconsistent");

	//squaredNorm(x1-x2)
	int size = x1.size();
	double squaredNorm = 0.0;
	for (int i = 0; i < size; i++)
	{
		squaredNorm += (x1[i] - x2[i])*(x1[i] - x2[i]);
	}

	double res = exp(-kernerlSigma * squaredNorm);

	return res;
}

/**
* Main entry called from Matlab
* @param nlhs number of left-hand-side arguments
* @param plhs pointers to mxArrays in the left-hand-side
* @param nrhs number of right-hand-side arguments
* @param prhs pointers to mxArrays in the right-hand-side
* @usage [ result ] = st_svm_evalute(svs_feats, svs_beta, kernerl_sigma, x_feats)
*/
void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
    // Check the number of arguments
	if (nrhs != 4)
		mexErrMsgIdAndTxt("st_svm_evaluate:error", "Wrong number of arguments input");

	if (nlhs != 1)
		mexErrMsgIdAndTxt("st_svm_evaluate:error", "Wrong number of arguments output");
    
	//output
	plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double* pResult = (double*)mxGetPr(plhs[0]);
    
    if(mxGetM(prhs[0])==0)
    {
        mexPrintf("sv is empty\n");
        pResult[0] = 0.0;
        mexPrintf("restult is 0.0\n");
        return;
    }
    
	//check the input dimension
    if(mxGetM(prhs[2])!=1 || mxGetN(prhs[2])!=1)
    {
        mexErrMsgIdAndTxt("st_svm_evaluate:error", "Wrong dimension of kernerl_sigma input");
    }
    
    if(mxGetM(prhs[3])!=1 && mxGetN(prhs[3])!=1)
    {
        mexErrMsgIdAndTxt("st_svm_evaluate:error", "Wrong dimension of xFeats input");
    }
    
    // rename input for convinence
    const mxArray *pSvsFeatsArray = prhs[0];
    const mxArray *pSvsBetaArray = prhs[1];
    double kernerlSigma = ((double*)mxGetPr(prhs[2]))[0];
    double *pXFeats = (double*)mxGetPr(prhs[3]);
    
    // construct feats to vector
    int featsDimension = mxGetM(pSvsFeatsArray);
    vector<double> xFeats(pXFeats, pXFeats + featsDimension);
    
	double f = 0.0;

	double* pSvsFeats = (double*)mxGetPr(pSvsFeatsArray);
	double* pSvsBeta = (double*)mxGetPr(pSvsBetaArray);

	int svsNum = mxGetM(pSvsBetaArray);
	
	for (int i = 0; i < svsNum; i++)
	{
		vector<double> svxFeats(pSvsFeats + i*featsDimension, pSvsFeats + i*featsDimension + featsDimension);
		f += pSvsBeta[i] * st_svm_kernel_eval(svxFeats, xFeats, kernerlSigma);
	}

	pResult[0] = f;
    return;
}

