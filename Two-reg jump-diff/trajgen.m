function y=trajgen(n)
load HH_6.dat
pp=HH_6;
aa=size(pp);
n_tot=aa(1);
TT=smooth(pp,0.1,'loess');
p=pp(n_tot-2191:n_tot-365);
T=TT(n_tot-2191:n_tot-365);
x=p-T;
x(1)=pp(1)-TT(1);
%a=size(p);
%nobs=a(1);
b=[0.0491    0.0220    0.1447    0.0525    0.0917    0.2500    0.9076    0.9155];
%b=[0.0495    0.0215    0.1489    0.0563    0.0456    0.3607    0.8920    0.8990];
for g=1:10000
nk(1)=2;    
for j=1:700
for i=nk(j):n
    if rand < b(7)
       x(i)=(1-b(1))*x(i-1)+b(2)*randn;
   else 
       if rand < 1-b(5)
        x(i)=(1-b(3))*x(i-1)+b(4)*randn;
        else
        x(i)=(1-b(3))*x(i-1)+b(4)*randn+b(6)*randn;
        end
   break
end
end
t=i;  
for m=t+1:n
    if rand < b(8)
        if rand < 1-b(5)
        x(m)=(1-b(3))*x(m-1)+b(4)*randn;
        else
        x(m)=(1-b(3))*x(m-1)+b(4)*randn+b(6)*randn;
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