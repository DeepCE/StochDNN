function f=last(b)
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
for k=1:nobs-1
    y0(k)=(1/sqrt(2*pi*b(2)^2)*exp(-1/(2*b(2)^2)*(x(k+1)-x(k)+b(1)*x(k))^2));
    ys(k)=((1-b(5))/sqrt(2*pi*b(4)^2)*exp(-1/(2*b(4)^2)*(x(k+1)-x(k)+b(3)*x(k))^2)....
    +b(5)/sqrt(2*pi*(b(4)^2+b(6)^2))*exp(-1/(2*(b(4)^2+b(6)^2))*(x(k+1)-x(k)+b(3)*x(k))^2));
end
P=[b(7),1-b(8);1-b(7),b(8)];
csi=[0;1];
for k=1:nobs-1
    lik(k)=csi(1)*y0(k)+csi(2)*ys(k);
    if lik(k)==0
      lik(k)=0.01;
   else    
    csi=P*[csi(1)*y0(k)/lik(k);csi(2)*ys(k)/lik(k)];
end
end
f=-sum(log(lik))