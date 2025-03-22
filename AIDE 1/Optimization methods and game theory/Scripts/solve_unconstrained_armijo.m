function [v,x] = solve_unconstrained_armijo(f, starting_point, gamma, t_bar, alpha)
    %SOLVE_QUADRATIC Solve an unconstrained problem with the
    % armijo inexact method

    vars = argnames(f);
    df = gradient(f, vars);
    
    x = starting_point;

    i = 0;
    while true
        i = i + 1;
        v = double(subs(f,vars,x.'));
        d = -double(subs(df,vars,x.'));
        disp(norm(d))
        if norm(d) < 1e-3
            break
        end
        t = t_bar;
        while double(subs(f,vars,(x + t*d).')) > double(subs(f,vars,x.')) - alpha * t * (d.' * d)
            t = gamma * t;
        end
        x = x + t*d;
    end
end