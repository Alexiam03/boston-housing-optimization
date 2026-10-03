function y = forward(A, X, x)
    Z = activation(A * X);
    y = Z * x;
end
