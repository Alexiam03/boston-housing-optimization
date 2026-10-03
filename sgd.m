function [X, x, history] = sgd(A, e, X, x, lr, batch_size, max_iter, tol)

    N = size(A, 1);
    history.loss = zeros(max_iter, 1);
    history.grad_norm = zeros(max_iter, 1);
    tic;

    for k = 1:max_iter

        % alegem aleator un batch de exemple
        idx = randperm(N, batch_size);
        A_batch = A(idx, :);
        e_batch = e(idx);

        % forward pe toate datele
        y_all = forward(A, X, x);
        L = loss(y_all, e);

        % gradient doar pe batch
        [gX, gx] = gradient(A_batch, e_batch, X, x);

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

        % update SGD
        X = X - lr * gX;
        x = x - lr * gx;

    end

    history.time = toc;
    history.iterations = length(history.loss);

end