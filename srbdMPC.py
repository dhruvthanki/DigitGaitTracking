import casadi as ca

class RigidBodyDynamics:
    def __init__(self):
        self.opti = ca.Opti()  # Create an Opti instance
        
        # Define constants
        self.m = 1.0  # Mass of the rigid body
        self.I = ca.DM.eye(3)  # Moment of inertia matrix, assuming a uniform sphere
        
        # Define state variables
        self.x = self.opti.variable(3)  # Position (x, y, z)
        self.v = self.opti.variable(3)  # Linear velocity
        self.q = self.opti.variable(4)  # Quaternion (orientation)
        self.omega = self.opti.variable(3)  # Angular velocity
        
        # Define control inputs
        self.F = self.opti.variable(3)  # External force
        self.Tau = self.opti.variable(3)  # External torque
        
        self.dt = 0.01  # Time step
        
    def setup_dynamics_and_objective(self):
        # Dynamics equations
        dxdt = self.v
        dvdt = self.F / self.m
        omega_quat = ca.vertcat(0, self.omega)  # Quaternion with zero real part
        
        # Quaternion derivative
        dqdt = 0.5 * ca.mtimes(ca.vertcat(
            ca.horzcat(self.q[3], -self.q[2], self.q[1], self.q[0]),
            ca.horzcat(self.q[2], self.q[3], -self.q[0], self.q[1]),
            ca.horzcat(-self.q[1], self.q[0], self.q[3], self.q[2]),
            ca.horzcat(-self.q[0], -self.q[1], -self.q[2], self.q[3])
        ), omega_quat)
        
        domegadt = ca.inv(self.I) @ (self.Tau - ca.cross(self.omega, self.I @ self.omega))
        
        # Integration for objective function
        x_next = self.x + self.dt * dxdt
        v_next = self.v + self.dt * dvdt
        q_next = self.q + self.dt * dqdt
        omega_next = self.omega + self.dt * domegadt
        
        # Objective function
        self.opti.minimize(
            ca.sum1(x_next**2) + 
            ca.sum1(v_next**2) + 
            ca.sum1((ca.norm_2(q_next) - 1)**2) + 
            ca.sum1(omega_next**2)
        )
    
    def solve(self):
        # Setup solver
        self.opti.solver("ipopt")
        
        # Solve the problem
        sol = self.opti.solve()
        
        # Extract and print the solution
        x_sol = sol.value(self.x)
        v_sol = sol.value(self.v)
        q_sol = sol.value(self.q)
        omega_sol = sol.value(self.omega)
        
        print("Solution:")
        print("Position:", x_sol)
        print("Velocity:", v_sol)
        print("Orientation (quaternion):", q_sol)
        print("Angular velocity:", omega_sol)

# Example usage:
if __name__ == "__main__":
    dynamics = RigidBodyDynamics()
    dynamics.setup_dynamics_and_objective()
    dynamics.solve()
