# from setuptools import setup
#
# setup(name='gym_pybullet_drones',
#     version='0.5.2',
#     install_requires=['numpy', 'Pillow', 'matplotlib', 'cycler', 'gym', 'pybullet', 'stable_baselines3', 'ray[rllib]']
# )


from setuptools import setup

setup(
    name='gym_pybullet_drones',
    version='0.6.0',
    packages=['gym_pybullet_drones'],  # 明确指定只安装这个包
    install_requires=[
        'numpy',
        'Pillow',
        'matplotlib',
        'cycler',
        'gym',
        'pybullet',
        'stable-baselines3',
        'ray'  # ray[rllib] 建议手动安装，避免中括号解析错误
    ],
)
