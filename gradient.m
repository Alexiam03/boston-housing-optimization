function [gX, gx] = gradient(A, e, X, x)
 N = size(A, 1);

%forward
U = A * X;
Z = U .* cos(U);
y = Z * x;

%calculam eroarea
r = y - e;

%gradientul fata de x
gx = (1/N) * (Z' * r);

%derivata activarii
gp = cos(U) - U .* sin(U);

%propagare eroare
delta = (1/N) * (r * x') .* gp;

%gradientul fata de X
gX = A' * delta;

end