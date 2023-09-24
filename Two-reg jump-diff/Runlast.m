clear
for g=1:10000
b=[rand,rand,rand,rand, 0.0456    0.3607,rand,rand];
lb=[0,0.001,0,0.001, 0.0456    0.3607,0,0];
ub=[inf,inf,inf,inf, 0.0456    0.3607,1,1];
options=optimset('MaxFunEvals',1e22)
[v,ll,ar,af,ag,grad,hess]=fmincon('last',b,[],[],[],[],lb,ub,[],options)
if last(v) < -3050
    break
    else
    end
end
C=sqrt(inv(hess));
D=diag(C)'
%SC=2*ll+length(b)*log(nobs)
v
