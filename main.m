data = readtable('BostonHousing.csv');
% sterge randurile care au valori lipsa
data = rmmissing(data);

A = data{:, 1:end-1};   % toate coloanele in afara de ultima
e = data{:, end};       % ultima coloana = preturile caselor

%randomizam datele 
N = size(A, 1);
i = randperm(N);

%impartim datele in cele de antrenare (80%) si cele pentru testare (20%)
train_i = i(1: round(0.8*N));
test_i = i(round(0.8*N) + 1: end);

A_train = A(train_i, :);
e_train = e(train_i);

A_test = A(test_i, :);
e_test = e(test_i);

% normalizare date folosind media si deviatia standard de pe train
mu = mean(A_train);
sigma = std(A_train);

A_train = (A_train - mu) ./ sigma;
A_test  = (A_test - mu) ./ sigma;

%adaugam bias
A_train = [A_train ones(size(A_train,1),1)];
A_test  = [A_test  ones(size(A_test,1),1)];

% numar neuroni strat ascuns
m = 10;

% dimensiune input (cu bias)
n_bar = size(A_train, 2);

% initializare ponderi
X = 0.01 * randn(n_bar, m);
x = 0.01 * randn(m, 1);

%% metoda 1: Momentum
lr = 1e-3;
beta = 0.9;
max_iter = 5000;
tol = 1e-6;

% test loss
y_train = forward(A_train, X, x);
L_train = loss(y_train, e_train);

disp("Loss initial pe train:");
disp(L_train);

[X_mom, x_mom, hist_mom] = momentum(A_train, e_train, X, x, lr, beta, max_iter, tol);

disp("Loss final momentum:");
disp(hist_mom.loss(end));

disp("Numar iteratii momentum:");
disp(hist_mom.iterations);

disp("Timp momentum:");
disp(hist_mom.time);

%% metoda 2: SGD
lr_sgd = 1e-3;
batch_size = 32;
max_iter_sgd = 5000;
tol_sgd = 1e-6;

[X_sgd, x_sgd, hist_sgd] = sgd(A_train, e_train, X, x, lr_sgd, batch_size, max_iter_sgd, tol_sgd);

disp("Loss final SGD:");
disp(hist_sgd.loss(end));

disp("Numar iteratii SGD:");
disp(hist_sgd.iterations);

disp("Timp SGD:");
disp(hist_sgd.time);

%% Grafice
% MOMENTUM 
figure;
plot(hist_mom.loss, 'b', 'LineWidth', 1.5);
xlabel('Iteratii');
ylabel('Loss');
title('Evolutia functiei obiectiv - Momentum');
grid on;

figure;
plot(hist_mom.grad_norm, 'b', 'LineWidth', 1.5);
xlabel('Iteratii');
ylabel('Norma gradientului');
title('Evolutia normei gradientului - Momentum');
grid on;

% SGD
figure;
plot(hist_sgd.loss, 'r', 'LineWidth', 1.5);
xlabel('Iteratii');
ylabel('Loss');
title('Evolutia functiei obiectiv - SGD');
grid on;

figure;
plot(hist_sgd.grad_norm, 'r', 'LineWidth', 1.5);
xlabel('Iteratii');
ylabel('Norma gradientului');
title('Evolutia normei gradientului - SGD');
grid on;

% loss comparativ
figure;
plot(hist_mom.loss, 'b', 'LineWidth', 1.5);
hold on;
plot(hist_sgd.loss, 'r', 'LineWidth', 1.5);
xlabel('Iteratii');
ylabel('Loss');
title('Comparatie Loss in functie de iteratii');
legend('Momentum', 'SGD');
grid on;

% gradient comparativ
figure;
plot(hist_mom.grad_norm, 'b', 'LineWidth', 1.5);
hold on;
plot(hist_sgd.grad_norm, 'r', 'LineWidth', 1.5);
xlabel('Iteratii');
ylabel('Norma gradientului');
title('Comparatie norma gradientului');
legend('Momentum', 'SGD');
grid on;

%% Evaluare date

% evaluare momentum
y_test_mom = forward(A_test, X_mom, x_mom);

MSE_mom = mean((y_test_mom - e_test).^2);
R2_mom = 1 - sum((e_test - y_test_mom).^2) / sum((e_test - mean(e_test)).^2);

disp("MSE momentum:");
disp(MSE_mom);

disp("R2 momentum:");
disp(R2_mom);

% evaluare SGD
y_test_sgd = forward(A_test, X_sgd, x_sgd);

MSE_sgd = mean((y_test_sgd - e_test).^2);
R2_sgd = 1 - sum((e_test - y_test_sgd).^2) / sum((e_test - mean(e_test)).^2);

disp("MSE SGD:");
disp(MSE_sgd);

disp("R2 SGD:");
disp(R2_sgd);