function [ mos, ypre,bayta ehat,J ] = LogisticFitting( mos,MetricValues )
% A 5 parameter non linearity(a logistic function with additive linear term constrained to be monotonic) is selected for the evaluation. 
% Changing the beta parameters will influence the fitting. 
beta(1) = max(mos);
beta(2) = min(mos);
beta(3) = median(MetricValues);
beta(4) = 0.1;
beta(5) = 0.1;
%fitting a curve using the data
[bayta ehat,J] = nlinfit(MetricValues,mos,@logistic,beta);
%given a metric value, predict the correspoing mos (ypre) using the fitted curve
[ypre junk] = nlpredci(@logistic,MetricValues,bayta,ehat,J);
end

