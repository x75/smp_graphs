"""test factory style class
"""
from smp_base.common import set_attr_from_dict

class mdl(object):
    defaults = {
        'mdls': {
            'm1': {'type': 1},
        },
    }
    def __init__(self, conf):

        # if len(conf['mdls']) > 1:
        # set_attr_from_dict(self, conf['mdls'])

        k1 = conf['mdls'].keys()[0]
        set_attr_from_dict(self, conf['mdls'][k1])

    def step(self, *args, **kwargs):
        print "%s.step type = %s" % (self.__class__.__name__, self.type)

# class create_mdls():
#     # def __init__(self, conf):
#     #     self.conf = conf
#     #     self()
#     def __call__(self, conf, *args, **kwargs):
#         self.conf = conf
#         for k, v in self.conf.items():
#             print "create_mdls.__call__ k = %s, v = %s" % (k, v)
#         return 'hallo'

def create_mdls(conf, *args, **kwargs):
    if conf.has_key('mdls'):
        
if __name__ == '__main__':

    conf1 = {
        'mdls': {
            'm1': {'type': 1},
        }
    }

    conf2 = {
        'mdls': dict([('m%d' % (k, ), {'type': v}) for k, v in zip(range(3), range(2, 5))])
    }

    print "conf1 = %s" % (conf1, )
    print "conf2 = %s" % (conf2, )

    m1 = mdl(conf1)
    m2 = mdl(conf2)

    m1.step()
    m2.step()
    
    m1_ = create_mdls(conf1)
    # m1_('bla')

    print "m1_ = %s" % (type(m1_))
