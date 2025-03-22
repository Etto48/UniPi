function [v,x] = solve_quadratic_unconstrained_conj_gradient(Q, c, starting_point)
    %SOLVE_QUADRATIC Solve a quadratic unconstrained problem with the
    % conjugate gradient method

    Q = (Q + Q.')/2;
    
    function [v] = f(x)
        v = 0.5 * x.'*Q*x + c.'*x;
    end
    
    x = starting_point;

    i = 0;
    while true
        i = i + 1;
        v = f(x);
        g = (Q*x+c);
        if i > 1
            b = g.'*Q*d/(d.'*Q*d);
            d = -g+b*d;
        else
            d = -g;
        end
        if norm(d) < 1e-6
            break
        end
        t = -(g.'*d)/(d.'*Q*d);
        x = x + t*d;
    end
end