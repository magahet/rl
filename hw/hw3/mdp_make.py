import json


states = [
    {
        'id': 0,
        'actions': [
            {
                'id': 0,
                'transitions': [
                    {
                        'id': 0,
                        'probability': 1,
                        'reward': 0,
                        'to': 1
                    }
                ]
            },
            {
                'id': 1,
                'transitions': [
                    {
                        'id': 0,
                        'probability': 1,
                        'reward': 0,
                        'to': 2
                    }
                ]
            }
        ]
    }
]


for a in xrange(1, 27, 2):
    b = a + 1
    states.append({
        'id': a,
        'actions': [
            {
                'id': 0,
                'transitions': [
                    {
                        'id': 0,
                        'probability': 1,
                        'reward': 0,
                        'to': a + 2
                    }
                ]
            },
            {
                'id': 1,
                'transitions': [
                    {
                        'id': 0,
                        'probability': 1,
                        'reward': 0,
                        'to': b + 2
                    }
                ]
            }
        ]
    })
    states.append({
        'id': b,
        'actions': [
            {
                'id': 0,
                'transitions': [
                    {
                        'id': 0,
                        'probability': 0.5,
                        'reward': 0,
                        'to': a + 2
                    },
                    {
                        'id': 1,
                        'probability': 0.5,
                        'reward': 0,
                        'to': b + 2
                    }
                ]
            },
            {
                'id': 1,
                'transitions': [
                    {
                        'id': 0,
                        'probability': 1,
                        'reward': 0,
                        'to': b
                    }
                ]
            }
        ]
    })

states.append({
    'id': 27,
    'actions': [
        {
            'id': 0,
            'transitions': [
                {
                    'id': 0,
                    'probability': 1,
                    'reward': 1,
                    'to': 29
                }
            ]
        },
        {
            'id': 1,
            'transitions': [
                {
                    'id': 0,
                    'probability': 1,
                    'reward': 0,
                    'to': 27
                }
            ]
        }
    ]
})

states.append({
    'id': 28,
    'actions': [
        {
            'id': 0,
            'transitions': [
                {
                    'id': 0,
                    'probability': 1,
                    'reward': 0,
                    'to': 29
                }
            ]
        },
        {
            'id': 1,
            'transitions': [
                {
                    'id': 0,
                    'probability': 1,
                    'reward': 0,
                    'to': 28
                }
            ]
        }
    ]
})

states.append({
    'id': 29,
    'actions': [
        {
            'id': 0,
            'transitions': [
                {
                    'id': 0,
                    'probability': 1,
                    'reward': 0,
                    'to': 29
                }
            ]
        },
        {
            'id': 1,
            'transitions': [
                {
                    'id': 0,
                    'probability': 1,
                    'reward': 0,
                    'to': 29
                }
            ]
        }
    ]
})

print json.dumps({
    'gamma': 0.75,
    'states': states
})
