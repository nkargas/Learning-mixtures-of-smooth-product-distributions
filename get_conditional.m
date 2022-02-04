function y = get_conditional(t,E,A,N_points,V_size)
T        = mean(diff(E));
cdf_est  = [cumsum([0; A])' ones(1,N_points-V_size-1)];
t_est    = [E max(E) + (1:N_points-V_size-1)*T];
z        = pi/T* (t'*ones(1,length(t_est)) - ones(length(t),1)*t_est);

dd = cos(z)./z - sin(z)./(z.^2);
dd(isnan(dd))=0;
y  = max(pi/T*(dd)*cdf_est',1e-5);
end