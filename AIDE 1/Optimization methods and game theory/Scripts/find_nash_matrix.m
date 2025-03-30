function [x, y] = find_nash_matrix(C)
    % Find the Nash equilibrium for a 2-player matrix game
    % Inputs:
    % C : mxn matrix with the costs for player 1
    % Outputs:
    % x : mx1 vector with the probabilities for player 1
    % y : nx1 vector with the probabilities for player 2

    m = size(C, 1);
    n = size(C, 2);

    c = [zeros(m,1); 1];
    A = [C' -ones(n,1)];
    b = zeros(n,1);
    Aeq = [ones(1,m), 0];
    beq = 1;
    lowerbound = [zeros(m,1); -Inf];
    upperbound = [];

    [sol, ~, ~, ~, lambda] = linprog(c, A, b, Aeq, beq, lowerbound, upperbound);
    x = sol(1:m);
    y = lambda.ineqlin;
end