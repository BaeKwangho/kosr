from se.models.EnhanceModel import EnhanceModel

def build_model(conf):
    model_type = conf['setting']['model_type']
    device = conf['setting']['device']
    
    if model_type=='segan':
        from se.models import segan as model
    else:
        raise ValueError('se.model is not registered')
    
    model_conf = conf['se_model']
    
    generator = model.Generator(model_conf)
    discriminator = model.Discriminator(model_conf)
    
    return EnhanceModel(generator,discriminator).to(device)