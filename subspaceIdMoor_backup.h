#pragma once
#include "armadillo"

struct ss{
	ss();
	~ss();
	arma::mat A, B, C, D;
	arma::uword ny, nu, nx;

	/* Uses u as (nu, N) to get ysim as (ny, N)*/
	arma::mat ss::dsim(double Ts, arma::mat const &u);

private:
	void ss::setsizes();
};

struct iddata{
	iddata();
	~iddata();
	arma::mat y, u;
	std::string tags_y, tags_u;
};
class subspaceIdMoor
{
public:
	subspaceIdMoor();
	~subspaceIdMoor();

	/* Description:
		Make a block Hankel matrix with the data y 
		containing i block-rows and j columns*/
	arma::mat blkhank(arma::mat const &y, arma::uword i, arma::uword j);
	void buildRhsLhsMatrix(arma::mat const &gam_inv, arma::mat const &gamm_inv, arma::mat const &R_, 
		arma::uword i, arma::uword n, arma::uword ny, arma::uword nu, arma::mat &RHS, arma::mat &LHS);
	void buildNMatrix(arma::uword k, arma::mat const &M, arma::mat const &L1, arma::mat const &L2, arma::mat const &X,
		arma::uword i, arma::uword n, arma::uword ny, arma::mat &N);

	/* Use the simple equation to solve for X the Discrete Lyapunov Equation
	ref: https://en.wikipedia.org/wiki/Lyapunov_equation */
	bool simple_dlyap(arma::mat const &A, arma::mat const &Q, arma::mat &X);

	/* Solves the Forward Riccati equation:
       P = A P A' + (G - A P C') (L0 - C P C')^{-1} (G - A P C')'        
       Using the generalized eigenvalue decomposition of page 62       
       flag = 1 when the covariance sequence is not positive real*/
	bool solvric(arma::mat const &A, arma::mat const &G, arma::mat const &C, arma::mat const &L0,
		arma::mat &P);

	/* Solve for the Kalman gain (K) and the innovation covariance (R)
         The resulting model is of the form:
               x_{k+1} = A x_k + K e_k
                 y_k   = C x_k + e_k
              cov(e_k) = R*/
	bool g12kr(arma::mat const &A, arma::mat const &G, arma::mat const &C, arma::mat const &L0,
		arma::mat &K, arma::mat &R);

	/* Compute Dynamic Matrix with a known order n*/
	ss subidKnownOrder(arma::uword ny, arma::uword nu, arma::mat const &R, arma::mat const &Usvd, arma::vec const &singval,
		arma::uword i, arma::uword n);

	/* Get information related to the system order.
	SV method is the singular values -- ref_nC;
	CVA method is the principal angles (degree) -- ref_nC
	Rfactor and Umatrix matrixs to keep identification of dynamics matrix*/
	bool subid_order(arma::mat const &y, arma::mat const &u, arma::uword i, arma::mat &Rfactor, 
		arma::mat &Usvd, arma::vec &s);
};

