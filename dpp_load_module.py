    def load_from_file(self, model, file):#
        state_dict = torch.load(file, map_location=lambda storage, loc: storage)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:7] == 'module.':
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return model
