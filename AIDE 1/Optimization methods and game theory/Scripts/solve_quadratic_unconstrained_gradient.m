function [v,x] = solve_quadratic_unconstrained_gradient(Q, c, starting_point)
    %SOLVE_QUADRATIC Solve a quadratic unconstrained problem with the
    % gradient method

    Q = (Q + Q.')/2;
    
    function [v] = f(x)
        v = 0.5 * x.'*Q*x + c.'*x;
    end
    
    x = starting_point;

    i = 0;
    while true
        i = i + 1;
        v = f(x);
        d = -(Q*x+c);
        if norm(d) < 1e-6
            break
        end
        t = -(d.'*(-d))/(d.'*Q*d);
        x = x + t*d;
    end
end