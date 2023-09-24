clear
load HH_6.dat
pp=HH_6;
aa=size(pp);
n_tot=aa(1);
TT=smooth(pp,0.1,'loess');
p=pp([n_tot-2191:n_tot-365]);
T=TT([n_tot-2191:n_tot-365]);
x=p-T;
a=size(p);
nobs=a(1);
nobs1=a(1);
for j=1:1000
y=trajgen(nobs1);
for i=1:nobs-1
    X(i)=x(i+1)-x(i);
end
for i=1:nobs1-1
    Y(i)=y(i+1)-y(i);
end
SX=sort(X);
SY=sort(Y);
f(j)=sum(abs(SX-SY))/(nobs-1);
%f(j)=sqrt(sum(abs(SX-SY).^2/(nobs-1)));
end
A=[mean(f) std(f)]