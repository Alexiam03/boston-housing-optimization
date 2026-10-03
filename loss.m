function L = loss(y, e)
    % calculeaza eroarea medie patratica:

    N = length(e);
    L = (1/(2*N)) * sum((y - e).^2);
end