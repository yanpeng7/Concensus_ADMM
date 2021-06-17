PROJ_NAME_PREFIX='proposed'

for num_iter in 100
do
  for gamma in 0.000001 0.00001 0.0001
  do
    for rho in 0.000001 0.00001 0.0001
    do
        for tau in 0.000001 0.00001 0.0001
        do
          for lambda_TV in 0.000001 0.00001 0.0001 0.001 0.01
          do
            for num_iter_TV in 30
            do
              PROJ_NAME=$PROJ_NAME_PREFIX'_num_iter='$num_iter'_gamma='$gamma'_rho='$rho'_tau='$tau'_lambda_TV='$lambda_TV'_num_iter_TV'$num_iter_TV
              python main.py --setting.proj_name=$PROJ_NAME --method.proposed.num_iter=$num_iter --method.proposed.gamma=$gamma --method.proposed.rho=$rho --method.proposed.tau=$tau --method.proposed.lambda_TV=$lambda_TV --method.proposed.num_iter_TV=$num_iter_TV
            done
          done
        done
    done

  done
done