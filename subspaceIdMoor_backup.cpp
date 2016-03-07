#include "subspaceIdMoor.h"
#include <cassert>
using namespace std;
using namespace arma;

iddata::iddata(){}
iddata::~iddata(){}

ss::ss(){}
ss::~ss(){}
void ss::setsizes(){
	ny = C.n_rows;
	nu = B.n_cols;
	nx = A.n_rows;
}
arma::mat ss::dsim(double Ts, arma::mat const &u){
	setsizes();
	mat ysim(ny, u.n_cols);
	mat xsim;
	xsim = zeros(nx, 1);
	ysim.col(0) = C * xsim;
	for (uword i = 1; i < u.n_cols; i++){ /*simple euler*/
		//xsim = xsim + Ts*(A*xsim + B*u.col(i));
		xsim = (A*xsim + B*u.col(i));
		ysim.col(i) = C * xsim + D * u.col(i);
	}
	return ysim;
}

subspaceIdMoor::subspaceIdMoor(){}
subspaceIdMoor::~subspaceIdMoor(){}
arma::mat subspaceIdMoor::blkhank(arma::mat const &y, uword i, uword j){
	assert(y.n_rows < y.n_cols);
	uword ny = y.n_rows;
	uword N = y.n_cols;
	if (j > N - i + 1)
		cerr << ("blkHank: j too big") << endl;
	mat H(ny*i, j);
	for (uword k = 0; k < i; k++)
		H.rows(k*ny, (k + 1)*ny - 1) = y.cols((k), k + j - 1);
	//H.save("H.dat", raw_ascii);

	return H;
}
bool subspaceIdMoor::subid_order(arma::mat const &y, arma::mat const &u, arma::uword i, arma::mat &Rfactor,
	arma::mat &Usvdmat, arma::vec &s){
	assert(u.n_rows < u.n_cols);
	assert(y.n_rows < y.n_cols);
	uword ny = y.n_rows;
	uword nu = u.n_rows;

	/*Optional Weighting flag:*/
	string Wstr = "SV";

	/* Dynamic System type */
	int ds_flag = 1;
	if (u.n_elem == 0)
		ds_flag = 2; /*stochastic*/
	else
		ds_flag = 1;

	/*General Checkings:*/
	assert(u.n_cols == y.n_cols);
	if ((y.n_rows - 2 * i + 1) < (2 * ny*i))
		cerr << "Not enough data points" << endl;

	/*Check the weight to be used ?*/

	// Determine the number of columns in the Hankel matrices
	uword j = y.n_cols - 2 * i + 1;

	/*Build output Block Hankel*/
	mat Y = blkhank(y / (double)sqrt(j), 2 * i, j);
	Y.save("Y.dat", raw_ascii);

	/*Build input block Hankel*/
	mat U = blkhank(u / (double)sqrt(j), 2 * i, j);
	U.save("U.dat", raw_ascii);

	/* R factor*/
	mat UY = join_vert(U, Y);
	mat Q, R;
	qr(Q, R, UY.t());
	R = R.t();
	R.save("R.dat", raw_ascii); /*ignoring triu command.*/
	R = R.submat(0, 0, 2 * i*(nu + ny) - 1, 2 * i*(nu + ny) - 1);

	/* Begin algorithm:*/

	/* Step 1:*/
	uword mi2 = 2 * nu*i;
	mat Rf = R.rows((2 * nu + ny)*i, 2 * (nu + ny)*i - 1); /*Future Outputs*/
	mat Rp = join_vert(R.rows(0, nu*i - 1), R.rows(2 * nu*i, (2 * nu + ny)*i - 1)); /*Past In/Out*/

	mat Ru, Rfpa, Rfpb, Rfpaslv, Rfp;
	mat Rppa, Rppb, Rppslv, Rpp;
	if (ds_flag == 1){
		Ru = R.submat(nu*i, 0, 2 * nu*i - 1, mi2 - 1); /*Future outputs*/
		Rfpaslv = solve(Ru.t(), (Rf.cols(0, mi2 - 1).t())).t(); /*Perpendicular Future outputs  
																to do: indicate that B is triangular*/
		Rfpa = Rf.cols(0, mi2 - 1) - Rfpaslv * Ru;
		Rfpb = Rf.cols(mi2, 2 * (nu + ny)*i - 1);
		Rfp = join_horiz(Rfpa, Rfpb);
		Rppslv = solve(Ru.t(), (Rp.cols(0, mi2 - 1).t())).t();
		Rppa = Rp.cols(0, mi2 - 1) - Rppslv * Ru; /*Perpendicular Past*/
		Rppb = Rp.cols(mi2, 2 * (nu + ny)*i - 1);
		Rpp = join_horiz(Rppa, Rppb);
	}

	/* Oblique projection*/
	mat Ob, Obslv, pinvret, Rppt;
	if (ds_flag == 1){
		/*Funny rank check: it is needed to avoid deficienty rank warnnings*/
		if (norm(Rpp.cols((2 * nu + ny)*i - 2 * ny - 1, (2 * nu + ny)*i - 1), "fro") < 1e-10){
			Rppt = Rpp.t();
			pinvret = pinv(Rppt);
			Ob = (Rfp*pinvret.t()) * Rp;
		}
		else{
			Obslv = solve(Rpp.t(), Rfp.t()).t();
			Ob = Obslv * Rp;
		}
	}

	/* Step 2: */
	/*   Compute the matrix WOW we want to take an SVD of
	W = 1 (SV), W = 2 (CVA)*/
	mat WOW, WOWslva, Qcva, Rcva, W1icva, Usvd, Vsvd;
	//vec s;
	if (ds_flag == 1){
		/*Extra projection of Ob on Uf perpendicular*/
		WOWslva = solve(Ru.t(), (Ob.cols(0, mi2 - 1)).t()).t();
		WOW = join_horiz(Ob.cols(0, mi2 - 1) - WOWslva * Ru, Ob.cols(mi2, 2 * (nu + ny)*i - 1));

		if (strcmp(Wstr.c_str(), "CVA") == 0){
			qr(Qcva, Rcva, Rf.t());
			W1icva = Rcva.submat(0, 0, ny*i, ny*i).t();
			WOW = solve(W1icva, WOW);
		}
		WOW.save("WOW.dat", raw_ascii);
		svd(Usvd, s, Vsvd, WOW);
		if (strcmp(Wstr.c_str(), "CVA") == 0)
			Usvd = W1icva * Usvd;
	}

	/* STEP 3*/
	/* Define the order from the singular values:*/
	mat ref_nC;
	if (strcmp(Wstr.c_str(), "CVA") == 0)
		ref_nC = real(acos(s)*180. / datum::pi); /*Principal angles in degree*/
	else{
		ref_nC = s;
	}
	Rfactor = R;
	Usvdmat = Usvd;
	return true;
}
void subspaceIdMoor::buildRhsLhsMatrix(arma::mat const &gam_inv, arma::mat const &gamm_inv, arma::mat const &R_,
	arma::uword i, arma::uword n, arma::uword ny, arma::uword nu, arma::mat &RHS, arma::mat &LHS){
	mat RhsUpper = join_horiz(gam_inv * R_.submat((2 * nu + ny)*i, 0, 2 * (nu + ny)*i - 1, (2 * nu + ny)*i - 1), zeros(n, ny));
	mat RhsLower = R_.submat(nu*i, 0, 2 * nu*i - 1, (2 * nu + ny)*i + ny - 1);
	RHS = join_vert(RhsUpper, RhsLower);
	mat LhsUpper = gamm_inv*R_.submat((2 * nu + ny)*i + ny, 0, 2 * (nu + ny)*i - 1, (2 * nu + ny)*i + ny - 1);
	mat LhsLower = R_.submat((2 * nu + ny)*i, 0, (2 * nu + ny)*i + ny - 1, (2 * nu + ny)*i + ny - 1);
	LHS = join_vert(LhsUpper, LhsLower);
}
void subspaceIdMoor::buildNMatrix(arma::uword k, arma::mat const &M, arma::mat const &L1, arma::mat const &L2, arma::mat const &X,
	arma::uword i, arma::uword n, arma::uword ny, arma::mat &N){
	mat Upper, Lower;
	Upper = join_horiz(M.cols((k - 1)*ny, ny*i - 1) - L1.cols((k-1)*ny, ny*i - 1), zeros(n, (k-1)*ny));
	Lower = join_horiz(-L2.cols((k - 1) * ny, ny*i - 1), zeros(ny, (k - 1)*ny));
	N = join_vert(Upper, Lower);
	if (k == 1)
		N.submat(n, 0, n + ny - 1, ny - 1) = eye(ny, ny) + N.submat(n, 0, n + ny - 1, ny - 1);
	N = N * X;
}
bool subspaceIdMoor::simple_dlyap(arma::mat const &A, arma::mat const &Q, arma::mat &X){
	mat kronProd = kron(A, A);
	mat I = eye(kronProd.n_rows, kronProd.n_cols);
	bool slvflg = solve(X, I - kronProd, vectorise(Q));

	/*Reshape vec to matrix:*/
	X.reshape(A.n_rows, A.n_rows);
	return slvflg;
}
bool subspaceIdMoor::solvric(arma::mat const &A, arma::mat const &G, arma::mat const &C, arma::mat const &L0,
	arma::mat &P){
	mat L0i = inv(L0);
	uword n = A.n_rows;
	/*Set up the matrices for the eigenvalue decomposition*/
	mat AA = join_vert(join_horiz(A.t()-C.t()*L0i*G.t(), zeros(n, n)), join_horiz(-G*L0i*G.t(), eye(n,n)));
	mat BB = join_vert(join_horiz(eye(n,n), -C.t()*L0i*C), join_horiz(zeros(n,n), A-G*L0i*C));

	/*Compute the eigenvalue decomposition*/
	cx_vec eigval;
	cx_mat eigvec;
	eig_pair(eigval, eigvec, AA, BB);

	/*If there's an eigenvalue on the unit circle => no solution*/
	vec abseval = abs(eigval);
	if (any(abs(abseval - ones(2 * n)) < 1e-9))
		return false; /* eigenvalue on the unit circle (return false)*/

	/* Sort e-vals by abs*/
	uvec isort = sort_index(abseval);

	/* Compute P*/
	cx_mat PauxNum = eigvec.rows(n, 2 * n - 1);
	cx_mat PauxDen = eigvec.rows(0, n - 1);
	cx_mat slvP;
	bool stat = solve(slvP, PauxDen.cols(isort.subvec(0, n - 1)).t(), PauxNum.cols(isort.subvec(0, n - 1)).t());
	slvP = slvP.t();
	P = real(slvP);
	return stat;
}
bool subspaceIdMoor::g12kr(arma::mat const &A, arma::mat const &G, arma::mat const &C, arma::mat const &L0,
	arma::mat &K, arma::mat &R){
	mat P;
	bool ricstat = solvric(A, G, C, L0, P);
	if (!ricstat)
		return false;
	R = L0 - C*P*C.t();
	bool Kstat = solve(K, R.t(), (G - A*P*C.t()).t());
	K = K.t();
	return Kstat;
}

ss subspaceIdMoor::subidKnownOrder(arma::uword ny, arma::uword nu,  arma::mat const &R, arma::mat const &Usvd, arma::vec const &singval,
	arma::uword i, arma::uword n){
	ss ssout;

	mat U1 = Usvd.cols(0, n - 1);
	/* STEP 4 in Subspace Identification*/
	/*Determine gam and gamm*/
	mat gam = U1 * diagmat(sqrt(singval.subvec(0, n - 1)));
	mat gamm = gam.rows(0, ny*(i - 1) - 1);
	mat gam_inv = pinv(gam); /*pseudo inverse*/
	mat gamm_inv = pinv(gamm); /*pseudo inverse*/

	/* STEP 5*/
	mat Rhs, Lhs;
	buildRhsLhsMatrix(gam_inv, gamm_inv, R, i, n, ny, nu, Rhs, Lhs);

	/* Solve least square*/
	mat solls;
	solls = solve(Rhs.t(), Lhs.t()).t();

	/* Extract system matrix:*/
	mat A, C;
	A = solls.submat(0, 0, n - 1, n - 1);
	C = solls.submat(n, 0, n + ny - 1, n - 1);
	mat res = Lhs - solls*Rhs;

	/* Recompute gamma from A and C:*/
	gam.zeros();
	gam.rows(0, ny - 1) = C;
	for (uword k = 2; k <= i; k++){
		gam.rows((k - 1)*ny, k*ny - 1) = gam.rows((k-2)*ny, (k-1)*ny - 1) * A;
	}
	gamm = gam.rows(0, ny*(i - 1) - 1);
	gam_inv = pinv(gam);
	gamm_inv = pinv(gamm);

	/* Recompute the states with the new gamma:*/
	buildRhsLhsMatrix(gam_inv, gamm_inv, R, i, n, ny, nu, Rhs, Lhs);

	/* STEP 6:*/
	/* Computing system Matrix B and D*/
	/*ref pag 125 for P and Q*/
	mat P = Lhs - join_vert(A, C) * Rhs.rows(0, n - 1);
	P = P.cols(0, 2*nu*i - 1);
	mat Q = R.submat(nu*i, 0, 2 * nu*i - 1, 2 * nu*i - 1); /*Future inputs*/

	/* Matrix L1, L2 and M as on page 119*/
	mat L1 = A * gam_inv;
	mat L2 = C * gam_inv;
	mat M = join_horiz(zeros(n, ny), gamm_inv);
	mat X = join_vert(join_horiz(eye(ny, ny), zeros(ny, n)), join_horiz(zeros(ny*(i-1), ny), gamm));

	/* Calculate N and the Kronecker products (page 126)*/
	mat N;
	uword kk = 1;
	buildNMatrix(kk, M, L1, L2, X, i, n, ny, N);
	mat totm = kron(Q.rows((kk-1)*nu, kk*nu - 1).t(), N);
	for (kk = 2; kk <= i; kk++){
		buildNMatrix(kk, M, L1, L2, X, i, n, ny, N);
		totm = totm + kron(Q.rows((kk - 1)*nu, kk*nu - 1).t(), N);
	}

	/* Solve Least Squares: */
	mat Pvec = vectorise(P);
	mat sollsq2 = solve(totm, Pvec);

	/*Mount B and D*/
	sollsq2.reshape(n + ny, nu);
	mat D = sollsq2.rows(0, ny - 1);
	mat B = sollsq2.rows(ny, ny+n - 1);

	/* STEP 7: Compute sys Matrix G, L0*/
	mat covv, Qs, Ss, Rs, sig, G, L0, K, Ro;
	if (norm(res) > 1e-10){ /*Determine QSR from the residuals*/
		covv = res*res.t();
		Qs = covv.submat(0, 0, n - 1, n - 1);
		Ss = covv.submat(0, n, n - 1, n + ny - 1);
		Rs = covv.submat(n, n, n+ny - 1, n+ny - 1);
		simple_dlyap(A, Qs, sig); /*solves discrete lyapunov matrix equation*/
		G = A*sig*C.t() + Ss;
		L0 = C*sig*C.t() + Rs;

		/* Determine K and Ro*/
		g12kr(A, G, C, L0, K, Ro);
	}

	/* Set to ss structure:*/
	ssout.A = A;
	ssout.B = B;
	ssout.C = C;
	ssout.D = D;
	//ssout.A = A; parei aqui -> add later the ones related with stochastic to ss.

	return ssout;
}
