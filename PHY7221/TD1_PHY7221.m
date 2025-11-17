% =========================================================================
% PHY7221 - Travail Dirigé 1
% September 2025
% =========================================================================

% Clear the algorithm
clear

% Simulation parameters ===================================================
km = 1e3; % Kilometer unit
r_earth = 6400*km; % radius of the earth, in km
r_sat = r_earth + 20200*km; % radius of satellite, in km
r_rx = r_earth + 1.3; % radius of the receiver, in km
theta_lim = acos(r_earth/(r_earth+r_sat)); % Limit angle of the sattelite's position, in rad
c = physconst('LightSpeed'); % speed of light, in m/s
t_approx = 0; % User offset, in sec
% User
coords_approx = [0 0 0]; % Approximate user coordinates (arbitrary)
% Sattelite coordinates (spherical)
coords_sats = [r_sat, theta_lim, pi/3; 
    r_sat, -theta_lim, pi/3; 
    r_sat, theta_lim/2, 5*pi/3;
    r_sat, -theta_lim/3, 5*pi/3];
n_sats = length(coords_sats);

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

% Q2 ======================================================================

% Calculating the user position

function [coords_user, time_user, H] = Position(c, coords_approx, coords_sats, t_user)
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
    % Add measurement noise (simulate imperfect satellite signals)
    sigma_rho = 10; % meters of noise
    rho_j = rho_j + sigma_rho * randn(size(rho_j));
    % Remaining intermediate variables
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
    time_user = tu + deltaX(4)/c;
end

% Position constraint by the radius of the Earth

% The rule is that the r component of the position calculated can't be smaller than
% r_earth. However, this shall only be used in a forced approximation,
% which is not the goal of this project.

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
[coords_user, time_user, H] = Position(c, coords_approx, coords_sats, t_approx);
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
user = scatter3(coords_user(1), coords_user(2), coords_user(3), 'b', 'filled', 'DisplayName', 'User');
approx = scatter3(coords_approx(1), coords_approx(2), coords_approx(3), 'r', 'filled', 'DisplayName', 'Approximation');

% Build legend labels including coordinates
user_label = sprintf('User (%.2e, %.2e, %.2e, %.2e)', coords_user, time_user);
approx_label = sprintf('Approximation (%.2e, %.2e, %.2e, %.2e)', coords_approx, t_approx);

% Final settings
axis equal; grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D model of the Earth');
legend([user approx], {user_label, approx_label});
hold off;

% Q3 ======================================================================

% Control the position of the satellites (with plot)

% The goal of this plot is to provide a visualization of the movement of
% satellites in their orbit and the influence of it on the error obtained
% in the positioning of an user with respect to the approximation.

function SatelliteControl()
    % Parameters
    km = 1e3;
    r_earth = 6400*km;
    r_sat = r_earth + 20200*km;
    theta_lim = acos(r_earth/(r_earth+r_sat)); % max angle for satellites
    c = physconst('LightSpeed');
    t_user = 0;
    coords_approx = [0 0 0];

    % Satellites (spherical: [r, theta, phi])
    coords_sats = [r_sat,  theta_lim,     pi/2; 
                   r_sat, -theta_lim,     2*pi/2; 
                   r_sat,  theta_lim/2,   3*pi/2;
                   r_sat, -theta_lim/3,   4*pi/2];
    n_sats = size(coords_sats,1);
    cartesian_sats = Cartesian(coords_sats);

    [coords_user, time_user, ~] = Position(c, coords_approx, coords_sats, t_user);
    coords_approx = Cartesian(coords_approx);

    % === Figure and Axes ===
    fig = figure('Position',[100 100 800 600]);
    ax = axes('Parent',fig,'Position',[0.1 0.25 0.8 0.7]); % space below for sliders

    % Earth
    [X,Y,Z] = sphere(75);
    X = r_earth*X; Y = r_earth*Y; Z = r_earth*Z;
    surf(ax,X, Y, Z, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    hold(ax,'on');
    colormap(ax,winter);
    axis(ax,'equal');
    axis(ax,'vis3d');
    xlabel(ax,'X'); ylabel(ax,'Y'); zlabel(ax,'Z');
    title(ax,'3D model of the Earth');
    grid(ax,'on');
    view(ax,3);

    % Points
    sats = scatter3(ax,cartesian_sats(:,1), cartesian_sats(:,2), cartesian_sats(:,3), ...
                    80, 'g', 'filled');
    user = scatter3(ax,coords_user(1), coords_user(2), coords_user(3), 'b', 'filled');
    approx = scatter3(ax,coords_approx(1), coords_approx(2), coords_approx(3), 'r', 'filled');

    % === Legend (dynamic) ===
    user_label = sprintf('User (%.2e, %.2e, %.2e, %.2e)', coords_user, time_user);
    approx_label = sprintf('Approximation (%.2e, %.2e, %.2e)', coords_approx);
    lgd = legend(ax, [user approx], {user_label, approx_label}, 'Location','best');

    % === Sliders and Labels ===
    theta_values = coords_sats(:,2)';
    slider_handles = gobjects(1,n_sats);
    label_handles  = gobjects(1,n_sats);
    for i = 1:n_sats
        ypos = 0.15 - (i-1)*0.05; % vertical placement
        slider_handles(i) = uicontrol('Style','slider', ...
            'Min',-theta_lim, 'Max',theta_lim, 'Value',theta_values(i), ...
            'Units','normalized','Position',[0.25 ypos 0.45 0.03], ...
            'Callback',@(src,~) PositionUpdate(src,i));
        label_handles(i) = uicontrol('Style','text', ...
            'String', sprintf('%.2fπ rad', theta_values(i)/pi), ...
            'Units','normalized','Position',[0.72 ypos 0.2 0.03]);
    end

    % === Callback ===
    function PositionUpdate(src, idx)
        % Update theta of satellite idx
        coords_sats(idx,2) = src.Value;

        % Update label (in multiples of pi)
        set(label_handles(idx), 'String', sprintf('%.2fπ rad', src.Value/pi));

        % Recompute Cartesian coordinates
        cartesian_sats = Cartesian(coords_sats);

        % Recompute user position
        [coords_user, time_user, ~] = Position(c, coords_approx, coords_sats, t_user);

        % Update plot
        set(sats,'XData',cartesian_sats(:,1), ...
                 'YData',cartesian_sats(:,2), ...
                 'ZData',cartesian_sats(:,3));
        set(user,'XData',coords_user(1), 'YData',coords_user(2), 'ZData',coords_user(3));

        % Update legend dynamically
        user_label = sprintf('User (%.2e, %.2e, %.2e, %.2e)', coords_user, time_user);
        approx_label = sprintf('Approximation (%.2e, %.2e, %.2e)', coords_approx);
        legend(ax, [user approx], {user_label, approx_label}, 'Location','best');
    end
end

SatelliteControl();

% Q4 ======================================================================

% Dilution of Precision (DOP) metrics

function [gdop, pdop, hdop, vdop, tdop] = DOP(H, c)
    % Least-squares solution matrix
    J = inv(H' * H);
    D = diag(J);

    % Metrics
    gdop = sqrt(D(1)^2 + D(2)^2 + D(3)^2 + D(4)^2);
    pdop = sqrt(D(1)^2 + D(2)^2 + D(3)^2);
    hdop = sqrt(D(1)^2 + D(2)^2);
    vdop = sqrt(D(3)^2);
    tdop = sqrt(D(4)^2) / c;
end

[gdop, pdop, hdop, vdop, tdop] = DOP(H, c);

% Comparative plots

% In Q3, the plot was done to observe the result of the moving satellite
% with an interactive visualization. It can still be used here, but since
% now the focus is on specific variables changing with the same variations
% as before, the results will be plotted in graphs.

function PositionAndTimeDOP()
    % Parameters
    km = 1e3;
    r_earth = 6400*km;
    r_sat = r_earth + 20200*km;
    theta_lim = acos(r_earth/(r_earth+r_sat)); % max angle for satellites
    c = physconst('LightSpeed');
    t_user = 0;
    coords_approx = [0 0 0];

    % Satellites (spherical: [r, theta, phi])
    coords_sats = [r_sat,  theta_lim,     pi/2; 
                   r_sat, -theta_lim,     2*pi/2; 
                   r_sat,  theta_lim/2,   3*pi/2;
                   r_sat, -theta_lim/3,   4*pi/2];
    n_sats = size(coords_sats,1);
    cartesian_sats = Cartesian(coords_sats);

    [coords_user, time_user, ~] = Position(c, coords_approx, coords_sats, t_user);
    coords_approx = Cartesian(coords_approx);


end

PositionAndTimeDOP();