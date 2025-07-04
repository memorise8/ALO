import os
import sys
import shutil
import argparse
from dotenv import dotenv_values
import json
from alm.__version__ import __version__
import requests
import time
from alm.utils import print_job_info, read_token_from_file, update_file_keys_in_json, zip_current_directory
import yaml
from alm.model import settings
import getpass

from alm.alo_llm_cli import ALC

settings.update()
experimental_plan = settings.experimental_plan
LCC_URL = experimental_plan.setting.ai_logic_deployer_url

##################### get info #####################

def __get_info(args):
    # alm get workspace_info
    # alm get image_info
    # alm get version

    token = read_token_from_file('access_token')
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # workspace info
    if args.workspace_info and not any([args.image_info, args.version]):
        workspace_id = check_ws(args.workspace_name)

        workspace_info_url = f"{LCC_URL}/api/v1/workspaces/{workspace_id}/info"

        # POST 요청 보내기
        workspace_info = requests.get(workspace_info_url, headers=headers)

        # 응답 처리
        if workspace_info.status_code == 200:
            print("Workspace info:", workspace_info.json())
        else:
            print("Error:", workspace_info.status_code, workspace_info.text)

    # image list
    elif args.image_info and not any([args.workspace_info, args.version]):
        version_rul = f"{LCC_URL}/api/v1/images/info"
        # Bearer 토큰이 필요할 경우 헤더에 추가

        # POST 요청 보내기
        images_list = requests.get(version_rul, headers=headers)

        if images_list.status_code == 200:
            print("Base images:", images_list.json())
        else:
            print("Error:", images_list.status_code, images_list.text)

    # version check
    else :
        version_rul = f"{LCC_URL}/api/v1/version"
        # POST 요청 보내기
        aic_version = requests.get(version_rul, headers=headers)
        if aic_version.status_code == 200:
            aic_version = aic_version.json()
            print("AIC Version: ", aic_version['aic']['versions'][0]['ver_str'])
        else:
            print("Error:", aic_version.status_code, aic_version.text)

def main():

    acli = ALC(LCC_URL, settings)

    if len(sys.argv) > 1:
        if sys.argv[-1] in ['-v', '--version']:
            print(__version__)
            return
        if sys.argv[1] in ['-h', '--help']:
            pass
        elif sys.argv[1] not in ['api', 'login', 'register', 'update', 'deploy', 'activate', 'deactivate', 'get']:  # v1 호환
            # ['run', 'history', 'register', 'update', 'delete', 'template', 'example', 'api', 'provision_create', 'login', 'apilist', 'provision_description', 'provision_delete']:  # v1 호환
            sys.argv.insert(1, 'api')
    else:
        sys.argv.insert(1, 'api')

    parser = argparse.ArgumentParser('alm', description='ALO(AI Learning Organizer)')
    parser.add_argument('-v', '--version', action='version', version=__version__)
    subparsers = parser.add_subparsers(dest='command')

    # ALO-LLM
    cmd_api = subparsers.add_parser('api', description='')

    # Auth
    cmd_login = subparsers.add_parser('login', description='Login')
    cmd_login.add_argument('--id', help='User id')
    cmd_login.add_argument('--password', help='User password')

    # Register (Solution)
    cmd_register = subparsers.add_parser('register', description='Register Service API')
    cmd_register.add_argument('-n', '--name', default = None, help='Service API name')
    cmd_register.add_argument('-w', '--workspace', default = None, help='Workspace name')
    cmd_register.add_argument('-i', '--image', default="python 3.12", help='Image info')
    cmd_register.add_argument('-v', '--version', default=None, help='Service API version')
    cmd_register.add_argument('-l', '--list', action="store_true", default = None, help='register list flag')
    cmd_register.add_argument('-u', '--update', type=str, default=None, help='register delete flag')
    # cmd_register.add_argument('-d', '--delete', action="store_true", default = None, help='register delete flag')
    cmd_register.add_argument('-d', '--delete', type=str, default=None, help='register delete flag')

    # cmd_register_list = subparsers.add_parser('register_list', description='Get list of Service API')
    # cmd_register_list.add_argument('-w', '--workspace' , default = None, help='Workspace name')

    cmd_register_update = subparsers.add_parser('update', description='Describe provision status')
    cmd_register_update.add_argument('-n', '--name', default=None, help='workspace name')
    cmd_register_update.add_argument('-w', '--workspace', default=None, help='workspace name')

    # cmd_register_delete = subparsers.add_parser('register_delete', description='Delete registered Service API')
    # cmd_register_delete.add_argument('-n', '--name', required=True, help='Service API name')
    # cmd_register_delete.add_argument('-w', '--workspace', default=None, help='workspace name')

    # cmd_register_version_delete = subparsers.add_parser('register_version_delete', description='Delete registered Service API')
    # cmd_register_version_delete.add_argument('-n', '--name', required=True, help='Service API name')
    # cmd_register_version_delete.add_argument('-w', '--workspace', default=None, help='workspace name')
    # cmd_register_version_delete.add_argument('-v', '--version', default=None, help='workspace name')

    # Deploy (Stream)
    cmd_deploy = subparsers.add_parser('deploy', description='Provision and deploy Service API')
    cmd_deploy.add_argument('-n', '--name', default=None, help='Service API name')
    cmd_deploy.add_argument('-w', '--workspace', default=None, help='workspace name')
    cmd_deploy.add_argument('-v', '--version', default=None, help='version name')
    cmd_deploy.add_argument('-l', '--list', action="store_true", default = None, help='register list flag')
    cmd_deploy.add_argument('-g', '--get', action="store_true", default = None, help='register list flag')
    cmd_deploy.add_argument('-d', '--delete', type=str, default=None, help='register delete flag')
    # cmd_deploy.add_argument('-d', '--delete', action="store_true", default = None, help='register list flag')
    cmd_deploy.add_argument('-u', '--update', action="store_true", default = None, help='register list flag')

    # cmd_deploy_create = subparsers.add_parser('deploy_create', description='Provision and deploy Service API')
    # cmd_deploy_create.add_argument('-n', '--name', required=True, help='Service API name')
    # cmd_deploy_create.add_argument('-w', '--workspace', default=None, help='workspace name')
    # cmd_deploy_create.add_argument('-v', '--version', default=None, help='version name')

    # cmd_deploy_list = subparsers.add_parser('deploy_list', description='Get list of Service API')
    # cmd_deploy_list.add_argument('-w', '--workspace', default=None, help='workspace name')

    # cmd_deploy_get = subparsers.add_parser('deploy_get', description='Get list of Service API')
    # cmd_deploy_get.add_argument('-w', '--workspace', default=None, help='workspace name')
    # cmd_deploy_get.add_argument('-n', '--name', default=None, help='Service API name')

    # cmd_deploy_delete = subparsers.add_parser('deploy_delete', description='Delete provision of Service API')
    # cmd_deploy_delete.add_argument('-n', '--name', required=True, help='Service API name')
    # cmd_deploy_delete.add_argument('-w', '--workspace', default=None, help='workspace name')

    # cmd_deploy_update = subparsers.add_parser('deploy_update', description='Delete provision of Service API')
    # cmd_deploy_update.add_argument('-n', '--name', required=True, help='Service API name')
    # cmd_deploy_update.add_argument('-w', '--workspace', default=None, help='workspace name')

    # Activate (StreamHistory)
    cmd_activate = subparsers.add_parser('activate', description='Describe provision status')
    cmd_activate.add_argument('-n', '--name', default=None,help='Service API name')
    cmd_activate.add_argument('-w', '--workspace', default=None, help='workspace name')
    cmd_activate.add_argument('-s', '--spec', default=None, help='spec info')
    cmd_activate.add_argument('-r', '--replicas', default=None, help='number of replicas')
    cmd_activate.add_argument('-l', '--list', action="store_true", default = None, help='register list flag')
    cmd_activate.add_argument('-g', '--get', action="store_true", default = None, help='register list flag')

    cmd_deactivate = subparsers.add_parser('deactivate', description='Describe provision status')
    cmd_deactivate.add_argument('-n', '--name', default=None,help='Service API name')
    cmd_deactivate.add_argument('-w', '--workspace', default=None, help='workspace name')

    # cmd_activate_list = subparsers.add_parser('activate_list', description='Describe provision status')
    # cmd_activate_list.add_argument('-n', '--name', required=True, help='Service API name')
    # cmd_activate_list.add_argument('-w', '--workspace', default=None, help='workspace name')

    # cmd_activate_get = subparsers.add_parser('activate_get', description='Describe provision status')
    # cmd_activate_get.add_argument('-n', '--name', required=True, help='Service API name')
    # cmd_activate_get.add_argument('-w', '--workspace', default=None, help='workspace name')

    # cmd_deactivate = subparsers.add_parser('deactivate', description='Describe provision status')
    # cmd_deactivate.add_argument('-n', '--name', required=True, help='Service API name')
    # cmd_deactivate.add_argument('-w', '--workspace', default=None, help='workspace name')

    cmd_get = subparsers.add_parser('get', description='Describe provision status')
    cmd_get.add_argument('-wn', '--workspace_name', default=None, help='workspace name')
    cmd_get.add_argument('-w', '--workspace_info', action="store_true", default=None, help='number of replicas')
    cmd_get.add_argument('-i', '--image_info', action="store_true", default = None, help='register list flag')
    cmd_get.add_argument('-v', '--version', action="store_true", default = None, help='register list flag')
    # # Workspace
    # cmd_workspaces_info = subparsers.add_parser('workspace_info', description='Describe provision status')
    # cmd_workspaces_info.add_argument('-w', '--workspace', default=None, help='workspace name')

    # # Misc
    # cmd_images = subparsers.add_parser('image_info', description='init of ALO-LLM')

    # cmd_version = subparsers.add_parser('version', description='init of ALO-LLM')

    args = parser.parse_args()
    # alm register
    # alm register -l, --list
    # register -d, --delete, -v
    # update
    # deploy -n, -v
    # deploy -l, --list
    # deploy -g, --get
    # deploy -d, --delete
    # activate
    # activate -l, --list
    # activate -g, --get
    # deactivate
    # alm get workspace_info
    # alm get image_info
    # alm get version
    commands = {
                'api': acli.api,
                'login' : acli.login,
                'register' : acli.register,
                # 'update': __update,
                'deploy': acli.deploy,
                'activate': acli.activate,
                'deactivate': acli.deactivate,
                'get': __get_info
                }
    commands[args.command](args)

