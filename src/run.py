import numpy as np
from tqdm import tqdm
from src.controller import Controller
from src.distmodel import DistModel
from src.quadcopter import Quadcopter
from src.trajectory_generation import TrajectoryGenerator
from src.disturbances import thermal, gaussians
from src.utils import classvar2file, data_out_dir, data2csv
from src import visualize

def main(SMOKE_TEST):
    np.random.seed(1)

    # Define Initial State
    pos0 = np.array((0, 0, 0))  # Initial x,y,z position of quadcopter
    yaw0 = 0  # initial yaw angle of quadcopter

    # Define Task
    n_rounds = 3  # Number of cycles
    t_per_round = 3  # should be > 3
    t_total = t_per_round * n_rounds  # Total time for all circles
    waypoints = np.concatenate(([pos0], np.tile(np.array([[0.1, 0, 0],
                                                          [0.1, 0.1, 0],
                                                          [0, 0.1, 0],
                                                          [0, 0, 0]]),
                                                (n_rounds, 1))))
    disturbance = gaussians
    # Create instances
    quad = Quadcopter(pos0, yaw0)
    dmodel = DistModel(2, 1,[-0.05, -0.05], [0.15, 0.15])  # None #
    traj_gen = TrajectoryGenerator(waypoints, t_total)
    ctrl = Controller(dmodel,traj_gen.get_desired_state(0))
    textend = -t_total + 3*quad.DT if SMOKE_TEST else 1
    t = np.arange(0, t_total + textend + quad.DT, quad.DT)
    x = np.zeros((t.size, 13))
    x_des = np.zeros((t.size, 3))
    x[0, :] = quad.state
    for i, ti in enumerate(tqdm(t[:-1])):

        # Get desired state, pass it to controller and store it
        desired_state = traj_gen.get_desired_state(ti)
        ctrl.set_desired_state(desired_state)
        x_des[i + 1, :] = desired_state['pos']

        # Simulate
        x[i + 1, :], w = quad.simulate(ctrl.run_ctrl, disturbance)
        if dmodel is not None:
            dmodel.add_data(x[i + 1, 0:2], w[:, 2], ctrl.get_last_epi())

    # Plot results
    if not SMOKE_TEST:
        visualize.x_vs_xd(t, x, x_des)
        if dmodel is not None:
            visualize.training_set(dmodel)
            # visualize.dis_model_surf(dmodel)
            mean_model, ale_model, epi_model, mean_true, ale_true = \
                visualize.dis_model_x(dmodel, disturbance, x[:, 0:2])
            # visualize.training_loss(dmodel)

        # Store data and parameters to file
        if dmodel is not None:
            fstr = '/dmodel_on_' + disturbance.__name__
        else:
            fstr = '/dmodel_off_'  + disturbance.__name__
        data_dir = data_out_dir(fstr)
        # utils.store_data(globals(), data_dir + 'variables')

        classvar2file(Quadcopter, data_dir + 'params_Quadcopter.json')
        classvar2file(traj_gen, data_dir + 'params_TrajectoryGenerator.json')
        if dmodel is not None:
            classvar2file(DistModel, data_dir + 'params_DistModel.json')
            data2csv(data_dir + 'dmodel.csv', xtr = dmodel.Xtr,
                           ytr = dmodel.Ytr, x_epi = dmodel.x_epi,
                           y_epi = dmodel.y_epi, loss = dmodel.loss,
                           mean_model = mean_model, ale_model = ale_model,
                           epi_model = epi_model, mean_true = mean_true,
                           ale_true = ale_true, x = x[:, 0:2])

        classvar2file(Controller, data_dir + 'params_Controller.json')
        data2csv(data_dir + 'simulation.csv',t=t, x = x[:,0:3], x_des = x_des)



if __name__ == "__main__":
    main(False)