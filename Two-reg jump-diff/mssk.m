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
for i=1:nobs-1
    h(i)=x(i+1)-x(i);
end
[mean(h),std(h),skewness(h),kurtosis(h)]
plot(x)