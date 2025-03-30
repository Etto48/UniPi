function [x,y] = find_nash_bimatrix(C1,C2,X0)
    % Find the Nash equilibrium for a bimatrix game
    % Inputs:
    % C1 : mxn matrix with the costs for player 1
    % C2 : mxn matrix with the costs for player 2
    % X0 : m+n+2x1 initial guess for the optimization
    % Outputs:
    % x : mx1 vector with the probabilities for player 1
    % y : nx1 vector with the probabilities for player 2

    [m,n] = size(C1);
    if m ~= size(C2,1) || n ~= size(C2,2)
        error("C1 and C2 must have the same dimensions");
    end

    H=[zeros(m,m) C1+C2 ones(m,1) zeros(m,1);
       C1'+C2' zeros(n,n) zeros(n,1) ones(n,1);
       ones(1,m) zeros(1,n+2);
       zeros(1,m) ones(1,n) 0 0];
    Ain = [C2' zeros(n,n) zeros(n,1) -ones(n,1);
        zeros(m,m) -C1 -ones(m,1) zeros(m,1)];
    bin = zeros(n+m,1);
    Aeq = [ones(1,m) zeros(1,n+2);
        zeros(1,m) ones(1,n) 0 0];
    beq = [1; 1];
    lowerbound = [zeros(m+n,1); -Inf; -Inf];
    upperbound = [ones(m+n,1); Inf; Inf];

    sol = fmincon(@(X) 0.5*X'*H*X, X0, Ain, bin, Aeq, beq, lowerbound, upperbound);
    x = sol(1:m);
    y = sol(m+1:m+n);
end