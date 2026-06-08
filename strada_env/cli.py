import argparse

def main():
    parser = argparse.ArgumentParser(prog='strada')
    subparsers = parser.add_subparsers(dest='command')

    # map editor
    editor = subparsers.add_parser('map-editor', help='Run the map editor')
    editor.add_argument('--window-size', type=int, nargs=2, default=(800, 800))
    editor.add_argument('--map-size', type=int, nargs=2, default=(18, 18))

    args = parser.parse_args()

    if args.command == 'map-editor':
        from strada_env import MapEditor
        me = MapEditor(
            window_size=args.window_size,
            map_size=args.map_size
        )
        me.run()
    elif args.command == 'cmd2':
        print("Reserved")
    else:
        parser.print_help()
