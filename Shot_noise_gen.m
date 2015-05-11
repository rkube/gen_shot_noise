function [sgnl,time] = Shot_noise_gen(g,l,N)
% This function returns a shot noise signal with intermittency parameter g,
% l is the relation between rise and decay time, and N is the number of
% bursts. The function returns the signal and its time steps.


t = 1:N;
dt = 1e-2;
K = N./(g.*dt);
tend = N/g;
time = dt:dt:tend;
Am = 1;
A = exprnd(Am,N,1);
tevent = rand(N,1)*tend;
tevent = sort(tevent);
kevent = round(tevent./dt);
trevent = kevent.*dt;  
twait = zeros(N-1,1);
for nn=1:N-1
    twait(nn) = trevent(nn+1)-trevent(nn);
end

sgnl = zeros(K,1);
S = zeros(K,1);
for i = 1:N
    S(1:kevent(i)) = A(i).*exp((-trevent(i) + dt:dt:0)./l);
    a = dt:dt:round((tend-trevent(i))./dt).*dt;
    S(kevent(i) + 1:end) = A(i).*exp(-(a)./(1-l));
    sgnl = sgnl + S;
end