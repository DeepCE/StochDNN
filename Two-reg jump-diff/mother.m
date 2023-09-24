clear
load HH_6.dat
pp=HH_6;
aa=size(pp);
n_tot=aa(1);
TT=smooth(pp,0.1,'loess');
p=pp([n_tot-2191:n_tot-365]);
T=TT([n_tot-2191:n_tot-365]);
ps=p-T;
a=size(p);
nobs=a(1);
b=[0.0491    0.0220    0.1447    0.0525    0.9076    0.9155];    
st_t=0.0715;
sk_t=-0.9708;
ku_t=46.6428;
%0.0917    0.2500 
a1_min=0.030;
a2_min=0.25;
sc1=0.01;
sc2=0.005;
lu1=100;
lu2=30;
lu3=30;
n=nobs;
for jj=1:lu1
for kk=1:lu2
    a1(jj)=a1_min+sc1*jj;
    a2(kk)=a2_min+sc2*kk;
for mm=1:lu3
x(1)=ps(1);
for g=1:10000
nk(1)=2;    
for j=1:700
for i=nk(j):n
    if rand < b(5)
       x(i)=(1-b(1))*x(i-1)+b(2)*randn;
   else 
       if rand < 1-a1(jj)
        x(i)=(1-b(3))*x(i-1)+b(4)*randn;
        else
        x(i)=(1-b(3))*x(i-1)+b(4)*randn+a2(kk)*randn;
        end
   break
end
end
t=i;  
for m=t+1:n
    if rand < b(6)
        if rand < 1-a1(jj)
        x(m)=(1-b(3))*x(m-1)+b(4)*randn;
        else
        x(m)=(1-b(3))*x(m-1)+b(4)*randn+a2(kk)*randn;
        end
   else
        x(m)=(1-b(1))*x(m-1)+b(2)*randn;
        break
end
end
s=m;
if s < n
    nk(j+1)=s+1;
else
    break
end
end
for i=1:n
z(i)=x(i);
end
if  max(z)-min(z) > 0
    if max(z)-min(z) < 3300000
        y=z;
    break
else
end
end
end
for i=1:n-1
    h(i)=x(i+1)-x(i);
end
st(mm)=std(h);
sk(mm)=skewness(h);
ku(mm)=kurtosis(h);
end
H(jj,kk)=mean(st);
Z(jj,kk)=std(st);
J(jj,kk)=mean(sk);
Y(jj,kk)=std(sk);
K(jj,kk)=mean(ku);
W(jj,kk)=std(ku);
if abs(H(jj,kk)-st_t)< Z(jj,kk)/2
    H1(jj,kk)=1;
else
    H1(jj,kk)=0;
end
if abs(J(jj,kk)-sk_t)< Y(jj,kk)
    J1(jj,kk)=1;
else
    J1(jj,kk)=0;
end
if abs(K(jj,kk)-ku_t)< W(jj,kk)
    K1(jj,kk)=1;
else
    K1(jj,kk)=0;
end
end
end    
MAG=H1+K1+J1;
i = 1;
for j=1:lu1
    for k=1:lu2
        if MAG(j,k)==3
            V(i,1) = a1_min+sc1*j;
            V(i,2) = a2_min+sc2*k;
            V(i,3) = H(j,k);
            V(i,4) = Z(j,k);
            V(i,5) = J(j,k);
            V(i,6) = Y(j,k);
            V(i,7) = K(j,k);
            V(i,8) = W(j,k);
            i = i + 1;
        end
    end
end
   V
