function [v,x] = solve_unconstrained_newton(f, starting_point)
    %SOLVE_QUADRATIC Solve an unconstrained problem with the
    % newton method

    vars = argnames(f);
    df = gradient(f, vars);
    hf = hessian(f, vars);
    
    x = starting_point;

    i = 0;
    while true
        i = i + 1;
        v = double(subs(f,vars,x.'));
        g = double(subs(df,vars,x.'));
        if norm(g) < 1e-6
            break
        end
        h = double(subs(hf,vars,x.'));
        d = linsolve(h,-g);
        x = x + d;
    end
end