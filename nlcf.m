function [ final_w, final_h, final_obj] = lcnmf( v, r, miu )
% Written by Yan Chen
%
%  v (m, n) : M (features) x N (samples) original matrix 
%           Numerical data only. 
%           Must be non negative. 
%           Not all entries in a row can be 0. If so, add a small constant to the 
%           matrix, eg.v+0.01*min(min(v)),and restart.
%           
%  r    : number of desired factors (rank of the factorization)
%
%  miu   : parameter of the constraint iterm
%
%  w    : M x r NMF factor
%  h    : r x N NMF factor

% test for negative values in v
if min(min(v)) < 0
    error('matrix entries can not be negative');
    return
end
if min(sum(v,2)) == 0
    zeroidx = find(sum(v,2) == 0);
    v(zeroidx, :) = [];
    fprintf('not all entries in a row can be zero');
end

% normalize
%maxv = max(max(v));
%v = v / maxv;

[m, n]=size(v);
% Maximum number of iterations (can be adjusted)
stopiter = 1000; 
obj = zeros(stopiter, 1);
% try times
trynum = 10;

for tn = 1 : trynum
    % Initialize random w and h
    w = abs(randn(m, r));
    h = abs(randn(r, n));
    
    % Start iteration
    for i = 1 : stopiter
        objsum = 0;
        % Construct lambda, objective, sum1 and sum2
        %for j = 1 : n
            %lambda = diag(h(:,j));
            %sum1 = sum1 + v(:, j) * ones(1, r) * lambda;
            %sum2 = sum2 + w * lambda;
            %objsum = objsum + miu * sum(sum( ((v(:,j)*ones(1,r) - w) * lambda.^0.5).^2 ));
        %end
        %clear lambda;
        sum1 = v * h';
        sum2 = w * diag(sum(h,2));
        
        % Calculate objective
        %if i ~= 1
        %    if mod(i, 100) == 0
        %        fprintf('Iterate %d times\n', i);
        %    end
        %end

        % Construct c and d
        %c = repmat(diag(v' * v)', r, 1);
        %d = repmat(diag(w' * w), 1, n);
        c = repmat(sum(v.^2), r, 1);
        d = repmat(sum(w.^2)', 1, n);

        % Update w and h
        h = h .* (2 * (miu+1) * (w'*v)) ./ (2*w'*w*h + miu*c + miu*d + 1e-9);
        w = w .* ((miu+1) * sum1) ./ (w*(h*h') + miu*sum2 + 1e-9);
        
        %obj(i) = sum(sum((v - w * h) .^ 2)) + objsum;    
        if i == stopiter
            for j = 1 : n
                lambda = diag(h(:,j));
                objsum = objsum + miu * sum(sum( ((v(:,j)*ones(1,r) - w) * lambda.^0.5).^2 ));
            end
            obj = sum(sum((v - w * h) .^ 2)) + objsum;
        end
    end
    if tn == 1
        final_obj = obj;
        final_w = w;
        final_h = h;
    else
        if obj < final_obj
            final_obj = obj;
            final_w = w;
            final_h = h;
        end
    end
end

end
