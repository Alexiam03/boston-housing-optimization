function [X, x, history] = momentum(A, e, X, x, lr, beta, max_iter, tol)

    vX = zeros(size(X));
    vx = zeros(size(x));

    history.loss = zeros(max_iter, 1);
    history.grad_norm = zeros(max_iter, 1);

    tic;

    for k = 1:max_iter

        y = forward(A, X, x);

        L = loss(y, e);

        % gradient
        [gX, gx] = gradient(A, e, X, x);

        % norma gradientului
        grad_norm = norm([gX(:); gx(:)]);

        % salvare istoric
        history.loss(k) = L;
        history.grad_norm(k) = grad_norm;

        % conditie de oprire
        if grad_norm < tol
            history.loss = history.loss(1:k);
            history.grad_norm = history.grad_norm(1:k);
            break;
        end

        % momentum update
        vX = beta * vX + gX;
        vx = beta * vx + gx;

        % actualizare parametri
        X = X - lr * vX;
        x = x - lr * vx;

    end

    history.time = toc;
    history.iterations = length(history.loss);

end