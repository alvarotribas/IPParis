% PHY7221 - Travail Dirigé 1
% September 2025

% Simulation parameters ===================================================
km = 1e3; % Kilometer unit
r_earth = 6400*km; % radius of the earth, in km
r_sat = r_earth + 20200*km; % radius of satellite, in km
r_rx = r_earth + 1.3; % radius of the receiver, in km
theta_lim = acos(r_earth/(r_earth+r_sat)); % Limit angle of the sattelite's position, in rad
c = physconst('LightSpeed'); % speed of light, in m/s
t_user = 1e-6; % User offset, in sec
% User
coords_approx = [r_rx 0.1 0.1]; % Approximate user coordinates (arbitrary)
% Sattelite coordinates (spherical)
coords_sats = [r_sat, theta_lim, pi/3; 
    r_sat, -theta_lim, pi/3; 
    r_sat, theta_lim/2, 5*pi/3;
    r_sat, -theta_lim/3, 5*pi/3];

% Changes of coordinate system ============================================

function spherical = Spherical(coords_cart) % Points are coords = [x1 y1 z1; x2 y2 z2; ...]
    % Cartesian
    x = coords_cart(:,1); y = coords_cart(:,2); z = coords_cart(:,3);

    % Spherical
    r = sqrt(x.^2 + y.^2 + z.^2);
    theta = acos(z ./ r);
    phi = atan2(y, x);
    % Result
    spherical = [r, theta, phi];
end

function cartesian = Cartesian(coords_sph) % Points are coords = [r1 theta1 phi1; r2 theta2 phi2; ...]
    % Spherical
    r = coords_sph(:,1); theta = coords_sph(:,2); phi = coords_sph(:,3);

    % Cartesian
    x = r .* sin(theta) .* cos(phi);
    y = r .* sin(theta) .* sin(phi);
    z = r .* cos(theta);
    % Result
    cartesian = [x,y,z];
end

% Calculating the user position ===========================================

function coords_user = Position(c, coords_approx, coords_sats, t_user)
    % Coordinates (initialization)
    xu = 0; yu = 0; zu = 0; tu = 0; % real user
    coords_approx = Cartesian(coords_approx); % approximate user in cartesian coordinates
    coords_sats = Cartesian(coords_sats); % satellites in cartesian coordinates
    xu_hat = coords_approx(:,1); yu_hat = coords_approx(:,2); zu_hat = coords_approx(:,3);
    xj = coords_sats(:,1)'; yj = coords_sats(:,2)'; zj = coords_sats(:,3)';
    %tu_hat = Cartesian(coords_approx(:,4)); % approximate

    % Intermediate variables
    rho_j = sqrt((xj - xu).^2 + (yj - yu).^2 + (zj - zu).^2) + c*t_user;
    rj_hat = sqrt((xj - xu_hat).^2 + (yj - yu_hat).^2 + (zj - zu_hat).^2);
    rhoj_hat = rj_hat + c*t_user;
    deltaRho = rhoj_hat - rho_j;

    a_xj = (xj - xu_hat) ./ rj_hat; 
    a_yj = (yj - yu_hat) ./ rj_hat; 
    a_zj = (zj - zu_hat) ./ rj_hat;
    a_j = [a_xj' a_yj' a_zj'];

    H = [a_j, ones(size(a_j,1),1)]; % H matrix has to be invertible in this case

    % Deltas between user and approximation
    deltaX = H\(deltaRho');

    % Real coordinates of the user
    coords_user = [xu - deltaX(1), yu - deltaX(2), zu - deltaX(3)]; % The ref equation did the sums
end

% Position constraint by the radius of the Earth
% The r component of the position calculated can't be smaller than r_earth
function coords_update = Constraint(coords_user, r_earth)
    % Into spherical in order to compare radii
    coords_user = Spherical(coords_user);
    
    % Condition
    if coords_user(1) < r_earth
        coords_user(1) = r_earth;
    end

    % Back to cartesian
    coords_update = Cartesian(coords_user);
end

% Functions and definitions ===============================================

% Transformation
cartesian = Cartesian(coords_sats);
x = cartesian(:,1); 
y = cartesian(:,2);
z = cartesian(:,3);

% User coordinates in cartesian
coords_user = Position(c, coords_approx, coords_sats, t_user);
coords_update = Constraint(coords_user, r_earth);
coords_approx = Cartesian(coords_approx);

% Plots Q2 ===================================================================

% Making the 3D plot 
[X,Y,Z] = sphere(75); % Use definition of sphere plot in matlab
X = r_earth*X; Y = r_earth*Y; Z = r_earth*Z; % Make points for plot

% Earth plot
figure;
surf(X, Y, Z, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
hold on;
colormap(winter);

% Sattelites plot
scatter3(x, y, z, 80, 'g', 'filled');

% Distances plot with corresponding distance texts
for i = 1:length(x)
    line([0, x(i)], [0, y(i)], [0, z(i)], 'Color', 'k', 'LineWidth', 0.2);
    rs = coords_sats(i, 1) - r_earth;
    text(x(i)/2, y(i)/2, z(i)/2, sprintf('%.2e', rs), ...
        'FontSize',10)
end

% User positions (approximate and calculated)
user = scatter3(coords_update(1), coords_update(2), coords_update(3), 'b', 'filled', 'DisplayName', 'User');
approx = scatter3(coords_approx(1), coords_approx(2), coords_approx(3), 'r', 'filled', 'DisplayName', 'Approximation');

text(coords_update(1), coords_update(2), coords_update(3), sprintf('(%.2e %.2e %.2e)', coords_update), ...
         'FontSize',9, 'Color','k', 'FontWeight','normal', ...
         'BackgroundColor','w', 'Margin',0.5);
text(coords_approx(1), coords_approx(2), coords_approx(3), sprintf('(%.2e %.2e %.2e)', coords_approx), ...
         'FontSize',9, 'Color','k', 'FontWeight','normal', ...
         'BackgroundColor','w', 'Margin',0.5);

% Final settings
axis equal; grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D model of the Earth');
legend([user approx], 'Location','best');
hold off;

% Plots Q3 ===================================================================