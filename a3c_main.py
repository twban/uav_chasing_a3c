# torch v2.0에서 비정상-->local network의 zero_grad()추가해서해결함.
# global-->cpu, local-->gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method
import numpy as np
import time
import a3c_env
import a3c_args as Args
import a3c_model


def train(global_models, args, rank, info_dict, train_loss_history, train_score_history, train_succ_fail_history):
    env = a3c_env.DroneTrackingENV(args, mode='TRAIN')
    local_models = []
    opt_globals = []
    opt_locals = []
    for i in range(4):
        local_model = a3c_model.ActorCritic(args).to(args.device)
        local_model.load_state_dict(global_models[i].state_dict())
        local_models.append(local_model)    
        opt_locals.append( optim.Adam(local_model.parameters(), lr=args.learning_rate) )
        opt_globals.append( optim.Adam(global_models[i].parameters(), lr=args.learning_rate) )
        

    ep = 0
    succ_cnt = 0
    while ep < args.max_train_ep and info_dict['train_stop']==False:
        ep += 1
        done = False
        # 1. Generate initial state
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        
        for i in range(4):
            local_models[i].reset_hidden(device=args.device)
            
        ep_score = np.zeros(4)
        ep_loss = np.zeros(4)
        ep_update_cnt = 0
        ep_step_cnt = 0

        # Start new episode
        while (not done) and (ep_step_cnt<args.fail_cnt):
            s_log_prob_lst, s_v_lst, s_r_lst, s_entropy_lst = [], [], [], []
            x_log_prob_lst, x_v_lst, x_r_lst, x_entropy_lst = [], [], [], []
            y_log_prob_lst, y_v_lst, y_r_lst, y_entropy_lst = [], [], [], []
            z_log_prob_lst, z_v_lst, z_r_lst, z_entropy_lst = [], [], [], []
            
            for t in range(args.update_interval):
                ep_step_cnt += 1                
                actions = [] #S, X, Y, Z
                # 1. Play
                probs, v = local_models[0](state.to(args.device)) #probs:torch.size[1, 5] 5=n_actions
                # local_models[0].detach_hidden()
                m = Categorical(probs)                
                action = m.sample() #torch.size[4]
                s_log_prob_lst.append( m.log_prob(action).unsqueeze(0) ) #torch.size[1, 1] #
                s_v_lst.append(v)                                         #torch.size[1, 1]
                s_entropy_lst.append(m.entropy().unsqueeze(0))          #torch.size[1, 1]
                actions.append(action.item())

                probs, v = local_models[1](state.to(args.device)) #probs:torch.size[1, 5] 5=n_actions
                # local_models[1].detach_hidden()
                m = Categorical(probs)                
                action = m.sample() #torch.size[4]
                x_log_prob_lst.append( m.log_prob(action).unsqueeze(0) ) #torch.size[1, 1] #
                x_v_lst.append(v)                                         #torch.size[1, 1]
                x_entropy_lst.append(m.entropy().unsqueeze(0))          #torch.size[1, 1]
                actions.append(action.item())

                probs, v = local_models[2](state.to(args.device)) #probs:torch.size[1, 5] 5=n_actions
                # local_models[2].detach_hidden()
                m = Categorical(probs)                
                action = m.sample() #torch.size[4]
                y_log_prob_lst.append( m.log_prob(action).unsqueeze(0) ) #torch.size[1, 1] #
                y_v_lst.append(v)                                         #torch.size[1, 1]
                y_entropy_lst.append(m.entropy().unsqueeze(0))          #torch.size[1, 1]
                actions.append(action.item())
                probs, v = local_models[3](state.to(args.device)) #probs:torch.size[1, 5] 5=n_actions
                # local_models[3].detach_hidden()
                m = Categorical(probs)                
                action = m.sample() #torch.size[4]
                z_log_prob_lst.append( m.log_prob(action).unsqueeze(0) ) #torch.size[1, 1] #
                z_v_lst.append(v)                                         #torch.size[1, 1]
                z_entropy_lst.append(m.entropy().unsqueeze(0))          #torch.size[1, 1]
                actions.append(action.item())

                # 2. Move
                next_state, rs, done  = env.move(actions)
                ep_score += np.array(rs)
                
                s_r_lst.append(rs[0])
                x_r_lst.append(rs[1])
                y_r_lst.append(rs[2])
                z_r_lst.append(rs[3])

                next_state = torch.FloatTensor(next_state).unsqueeze(0)          
                state = next_state
                if done:
                    break
            
            for local_model in local_models:
                local_model.detach_hidden()
            
            log_prob_lst_batch=[s_log_prob_lst, x_log_prob_lst, y_log_prob_lst, z_log_prob_lst]
            v_lst_batch=[s_v_lst, x_v_lst, y_v_lst, z_v_lst]
            r_lst_batch=[s_r_lst, x_r_lst, y_r_lst, z_r_lst]
            entropy_lst_batch=[s_entropy_lst, x_entropy_lst, y_entropy_lst, z_entropy_lst]
            losses = []


            # Start of update
            for i in range(4):    
                if done:
                    R = torch.zeros(1, 1) 
                else:
                    _, R = local_models[i](next_state.to(args.device), hidden_update=False)
                R = R.to(args.device)
                td_target_lst = []
                for r in reversed(r_lst_batch[i]):
                    R = r + args.gamma * R
                    td_target_lst.insert(0,R)

                td_targets  = torch.cat(td_target_lst, dim=0) #torch.size[update_interval, 4] #4=n_actions
                values      = torch.cat(v_lst_batch[i], dim=0)         #torch.size[update_interval, 4]
                log_probs   = torch.cat(log_prob_lst_batch[i], dim=0)  #torch.size[update_interval, 4]
                entropies   = torch.cat(entropy_lst_batch[i], dim=0)   #torch.size[update_interval, 4]
                advantages  = td_targets - values             #torch.size[update_interval, 4]
                
                # print(td_targets.shape, values.shape, log_probs.shape, advantages.shape)
                loss =  - log_probs * advantages.detach()\
                        + F.smooth_l1_loss(values, td_targets.detach())\
                        - args.entropy_coeff * entropies

                opt_locals[i].zero_grad()
                opt_globals[i].zero_grad()
                loss.mean().backward() # gradient of local network
                
                # copy the gradient of local to global
                for global_param, local_param in zip(global_models[i].parameters(), local_models[i].parameters()):
                    local_param.grad.data.clamp_(-1, 1)
                    global_param.grad = local_param.grad.cpu()

                # update the global using the copied gradient
                opt_globals[i].step()
                local_models[i].load_state_dict(global_models[i].state_dict())
                losses.append(loss.mean().item())

            # End of update
            ep_update_cnt += 1
            ep_loss += np.array(losses)
            
            #************ End of Update

        #* End of Episode While

        if done:
            succ_cnt += 1
            # print(f"    -Success[{rank}]  Ep:{ep+1}  Steps:{ep_step_cnt}  Dist:{env.dist_target:.1f}")

        ep_loss = ep_loss/ep_update_cnt
        train_loss_history.append(np.insert(ep_loss, 0, rank))
        train_score_history.append(np.insert(ep_score, 0, rank)) #Add process-id 
        train_succ_fail_history.append([rank, done])

        if (ep+1) % args.print_every == 0:
            ep_avg_score = np.array(train_score_history[-args.print_every:]).mean(axis=0)[1:] # The first i proc-id
            ep_avg_loss = np.array(train_loss_history[-args.print_every:]).mean(axis=0)[1:] # The first i proc-id
            
            print_str =f"    Train[{rank}] Ep:{ep+1}\tSucc:{succ_cnt}/{args.print_every}\tEp.Avg.Score:{np.round(ep_avg_score, 1)}\tEp.Avg.Loss:{np.round(ep_avg_loss,3)}"
            info_dict['train_status'] = print_str
            succ_cnt = 0

    # End of training While
    if info_dict['train_stop']==True:
        print_str =f"\tTrain[{rank}]\tStopped. Target Achieved"
    else:
        print_str =f"\tTrain[{rank}]\tFinished. Max Episodes"
        
    info_dict['train_status'] = print_str
    info_dict['train_done_cnt'] += 1



def test(global_models, args, info_dict):
    test_device = 'cuda'    
    env = a3c_env.DroneTrackingENV(args, mode='TEST')
    test_models = []
    for i in range(4):
        test_model = a3c_model.ActorCritic(args).to(test_device)
        test_model.load_state_dict(global_models[i].state_dict())
        test_models.append(test_model)
        
    ep = 0
    ep_step_cnt = 0
    ep_score = np.zeros(4)
    succ_history = []
    succ_cnt_old = 0
    with torch.no_grad():
        while info_dict['train_done_cnt'] < args.n_train_processes:
            ep += 1            
            state = env.reset()
            state = torch.FloatTensor(state).to(test_device).unsqueeze(0)
            for i in range(4):
                test_models[i].reset_hidden(device=test_device)
            done = False
            #** Start of Episode
            while (not done) and (ep_step_cnt<args.fail_cnt):
                ep_step_cnt += 1
                
                # Choose action
                actions = [] # S, X, Y, Z
                for i in range(4):
                    probs, _ = test_models[i](state) #probs:torch.size[1, 5] 5=n_actions
                    test_models[i].detach_hidden()
                    m = Categorical(probs)                
                    action = m.sample()
                    actions.append(action.item())
                # Move
                next_state, rs, done  = env.move(actions)
                ep_score += np.array(rs)
                
                next_state = torch.FloatTensor(next_state).to(test_device).unsqueeze(0)          
                state = next_state
                if done:
                    break
            #** End of Episode            
            succ_history.append(done)
            
            if ep%args.test_interval == 0:
                ep_score = ep_score/args.test_interval
                succ_cnt = np.array(succ_history).sum()
                print_str =f"Test Ep:{ep}\tSucc Rate:{succ_cnt}/{args.test_interval}\tEp.Avg.Score:{np.round(ep_score, 1)}"
                info_dict['test_status'] = print_str                
                if succ_cnt > succ_cnt_old:
                    for i in range(4):
                        torch.save(test_models[i].state_dict(), f"./chkpts/model_{i}_doppler0{int(args.doppler*10)}_b0{int(args.corr_coeff*10)}.pth")
                    succ_cnt_old = succ_cnt
                    print("\nModel Saved\n")

                for i in range(4):
                    test_models[i].load_state_dict(global_models[i].state_dict())
                
                ep_score = np.zeros(4)
                ep_step_cnt = 0
                succ_history = []


            # if  np.array(succ_history)[-args.stop_succ_cnt:].sum() >= args.stop_succ_cnt:                
            #     info_dict['train_stop'] = True
            #     time.sleep(1)
            #     info_dict['test_status'] = f"Target Achieved At {ep}. Training Early Terminated"

        info_dict['test_status'] = "Testing Done"


#*************************************************************8
#* reward: 이전 dst_target대비 개선 폭으로 정의
#* Test에서 avg.score가 0 근처로 나오는 것은 초반 움직임이 없으면 이전대비 변화가 없기 때문

if __name__ == '__main__':
    set_start_method('spawn')
    dopplers = [1.0]
    corr_coeffs = [0.1, 0.5, 0.9]
    for doppler in dopplers:
        for corr_coeff in corr_coeffs:
            print(f"\n\nDoppler={doppler}, Corr={corr_coeff}\n\n")
            ## CNN should be `False` !!!!
            args = Args.Args(cnn=False,\
                            initial_dist_target=100,\
                            initial_space_drones=5,\
                            doppler=doppler,\
                            corr_coeff=corr_coeff,\
                            n_rssi_samples=50,\
                            succ_thres_dist=5
                            )
            global_models = []
            for _ in range(4):
                global_model = a3c_model.ActorCritic(args)
                global_model.share_memory()
                global_models.append(global_model)
            s_time = time.time()

            manager = mp.Manager()    
            train_loss_history = manager.list()
            train_score_history = manager.list()
            train_succ_fail_history = manager.list()
            info_dict = manager.dict()
            info_dict['train_done_cnt'] = 0
            info_dict['train_status'] = None
            info_dict['test_status'] = None
            info_dict['train_stop'] = False
            
            processes = []
            for rank in range(args.n_train_processes + 1):  # + 1 for test process
                if rank == 0:
                    p = mp.Process(target=test, args=(global_models, args, info_dict))
                    pass
                else:
                    p = mp.Process(target=train, args=(global_models, args, rank, info_dict, train_loss_history, train_score_history, train_succ_fail_history))
            
                p.start()
                processes.append(p)

            while (info_dict['train_done_cnt'] < args.n_train_processes) or (info_dict['train_status'] is not None) or (info_dict['test_status'] is not None):
                if info_dict['train_status'] is not None:
                    print(info_dict['train_status'])
                    info_dict['train_status'] = None

                if info_dict['test_status'] is not None:
                    print(info_dict['test_status'])
                    info_dict['test_status'] = None
                time.sleep(0.01)

            for p in processes:
                p.join()
            
            path = f"./results/train_doppler0{int(doppler*10)}_b0{int(corr_coeff*10)}.npz"
            np.savez(path, doppler=doppler, corr_coeff=corr_coeff, loss = train_loss_history, ep_score = train_score_history, succ_fail = train_succ_fail_history )

            print(f"All Done. Time={time.time() - s_time:.1f}")