h = 0.1;
mu = 0.1;
Omega = linspace(0,0.5,500);
D = 4;
rp = 2.0;
R = zeros(2,length(Omega));

for i=1:length(Omega)
    [r, Htr, Gtr] = RK45(h, mu, Omega(i), D, 10^-13);
    Htr = Htr.'; Gtr = Gtr.';
    R(1,i) = Omega(i);
    R(2,i) = ( (-Gtr(end-1)*(r(end-1)-rp)+Htr(end-1)*(rp*Omega(i)-1))...
        /( Gtr(end-1)*(r(end-1)-rp)+Htr(end-1)*(rp*Omega(i)+1) ) )*(r(end-1)-rp)^(2*rp*Omega(i));
end


function v = V(r)
    v = 1 - 2./r;
end

function p = P(r, mu, Omega, D)
    rp = 2.0;
    num = mu*mu*( (D-2)-2*(rp./r).^(D-3)+(4-D)*(rp./r).^(2*D-6) )./(r.*V(r))+Omega*Omega...
        *( (D-2)+(2*D-7)*(rp./r).^(D-3) )/(r.*V(r))+( (D-2)-(rp./r).^(D-3) )*(rp./r).^(2*D-6);
    denom = -Omega^2 -V(r)*mu^2 + ( (D-3)^2 *(rp./r).^(2*D-6) )./(4*r.^2);

    p = num./denom;
end

function q = Q(r, mu, Omega, D)
    rp = 2.0;
    num = ( mu^2 +Omega^2 ./V(r) ).^2 +( Omega^2 ./(4*r.^2 .*V(r).^2) )...
        .*( 4*D-8-(8*D-16)*(rp/r).^(D-3) -(53-34*D+5*D^2)*(rp./r).^(2*D-6) )...
        +( mu^2 ./(4*r.^2 .*V(r)) ).*( 4*D-8-4*(3*D-7)*(rp./r).^(D-3)+(D^2 +2*D-11)...
        *(rp./r).^(2*D-6) )+( (D-3)^2 ./(4*r.^4 +V(r).^2) ).*(rp/r).^(2*D-6)...
        .*( (D-2)*(2*D-5)-(D-1)*(D-2).*(rp/r).^(D-3)+(rp./r).^(2*D-6) );
    denom = -Omega^2 -V(r)*mu^2 + ( (D-3)^2 *(rp/r).^(2*D-6) )./(4*r.^2);

    q = -num./denom;
end

function [rr, Htr, Gtr] = RK45(h, mu, Omega, D, rtol)
    c30 = 3/8;
    c31 = 3/32;
    c32 = 9/32;
    c40 = 12/13;
    c41 = 1932/2197;
    c42 = -7200/2197;
    c43 = 7296/2197;
    c51 = 439/216;
    c52 = -8;
    c53 = 3680/513;
    c54 = -845/4104;
    c61 = -8/27;
    c62 = 2;
    c63 = -3544/2565;
    c64 = 1859/4104;
    c65 = -11/40;
    cw1 = 25/216;
    cw3 = 1408/2565;
    cw4 = 2197/4104;
    cw5 = -1/5;
    cz1 = 16/135;
    cz3 = 6656/12825;
    cz4 = 28561/56430;
    cz5 = -9/50;
    cz6 = 2/55;
    ce1 = 1/360;
    ce3 = -128/4275;
    ce4 = -2197/75240;
    ce5 = 1/50;
    ce6 = 2/55;

    atol = 10^-13;
    alpha = 0.8;
    k = 0;
    i = 1;
    r1 = 200.0;
    r2 = 2.001;
    rr(1) = r1;
    ri = r1;
    htr0 = exp(-sqrt(mu^2 +Omega^2)*ri);
    gtr0 = -sqrt(mu^2 +Omega^2)*htr0;
    Htr(1,:) = htr0; Gtr(1,:) = gtr0;
    Hi = htr0; Gi = gtr0;

    lastit = 0;
    while lastit == 0
        %if ri - 1.1*h > r2
        %    h = r2 - ri;
        %    lastit = 1;
        %end
        if ri < r2
            break
        end

        K1 = h*Gi;
        L1 = h*P(ri,mu,Omega,D)*Gi+h*Q(ri,mu,Omega,D)*Hi;
        K2 = h*(Gi+0.25*L1);
        L2 = h*P(ri+0.25*h,mu,Omega,D)*(Gi+0.25*L1)+h*Q(ri+0.25*h,mu,Omega,D)*(Hi+0.25*K1);
        K3 = h*(Gi+c31*L1+c32*L2);
        L3 = h*P(ri+c30*h,mu,Omega,D)*(Gi+c31*L1+c32*L2)+h*Q(ri+c30*h,mu,Omega,D)...
            *(Hi+c31*K1+c32*K2);
        K4 = h*(Gi+c41*L1+c42*L2+c43*L3);
        L4 = h*P(ri+c40*h,mu,Omega,D)*(Gi+c41*L1+c42*L2+c43*L3)+h*Q(ri+c40*h,mu,Omega,D)...
            *(Hi+c41*K1+c42*K2+c43*K3);
        K5 = h*(Gi+c51*L1+c52*L2+c53*L3+c54*L4);
        L5 = h*P(ri+h,mu,Omega,D)*(Gi+c51*L1+c52*L2+c53*L3+c54*L4)...
            +h*Q(ri+h,mu,Omega,D)*(Hi+c51*K1+c52*K2+c53*K3+c54*K4);
        K6 = h*(Gi+c61*L1+c62*L2+c63*L3+c64*L4+c65*L5);
        L6 = h*P(ri+0.5*h,mu,Omega,D)*(Gi+c61*L1+c62*L2+c63*L3+c64*L4+c65*L5)...
            +h*Q(ri+0.5*h,mu,Omega,D)*(Hi+c61*K1+c62*K2+c63*K3+c64*K4+c65*K5);

        H_4 = Hi + h*(cw1*K1 + cw3*K3 + cw4*K4 + cw5*K5);
        G_4 = Gi + h*(cw1*L1 + cw3*L3 + cw4*L4 + cw5*L5);

        H_5 = Hi + h*(cz1*K1 + cz3*K3 + cz4*K4 + cz5*K5 + cz6*K6);
        G_5 = Gi + h*(cz1*L1 + cz3*L3 + cz4*L4 + cz5*L5 + cz6*L6);

        err = h*abs(ce1*K1 + ce3*K3 + ce4*K4 + ce5*K5 + ce6*K6);

        tol = rtol*abs(Hi) + atol;

        if err <= tol
            ri = ri - h;
            h = alpha*h*(tol/err)^0.4;
            i = i + 1;
            rr(i) = ri;
            Hi = H_5;
            Gi = G_5;
            Htr(i,:) = H_5;
            Gtr(i,:) = G_5;
            k=0;
        elseif k == 0
            h = alpha*h*(tol/err)^0.2;
            k = k + 1;
            lastit = 0;
        else 
            h = 0.5*h;
            lastit = 0;
        end

    end
end